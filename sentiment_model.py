# In this file, we will create the model that we will use to generate our embeddings
# Our model will be trained on the sentiment analysis task
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm

# Hyperparameters
batch_size = 64
#T = 2000 # Context length
max_iters = 2000
eval_interval = 20
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device', device)

# Embedding dimension
p = 1000
#----------

# For reproducibility
torch.manual_seed(1337)

# Load the IMDB datset
class Sentiment(Dataset):
    def __init__(self, split, tokenizer):
        # split is either train or test
        self.tokenizer = tokenizer
        #self.context_length = context_length
        ds = load_dataset("stanfordnlp/imdb")
        data = ds[split]
        self.prompts = data['text'] # list of str
        self.labels = torch.tensor(data['label'], dtype = torch.float) # tensor containing 0, 1

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # Tokenization
        tokens = tokenizer.encode(self.prompts[idx])
        n = torch.tensor(len(tokens), dtype = torch.float)
        x = torch.tensor(tokens, dtype= torch.long)
        y = self.labels[idx]
        return x.to(device), y.to(device), n.to(device)

# Collate function: that adds rows of different lengths to the same tensor
def collate_fn(batch):
    X = torch.nested.nested_tensor([x for (x, y, n) in batch])
    X.requires_grad = False
    Y = torch.stack([y for (x, y, n) in batch])
    Y.requires_grad = False
    N = torch.stack([n for (x, y, n) in batch]).view(len(batch))
    N.requires_grad = False
    return X, Y, N

# Tokenizer
tokenizer = tiktoken.get_encoding("o200k_base")
vocab_size = tokenizer.max_token_value

# Datasets 
train_data = Sentiment('train', tokenizer)
test_data = Sentiment('test', tokenizer)

# DataLoaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)
eval_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)
test_dataloader = DataLoader(test_data, batch_size= batch_size , shuffle=True, collate_fn= collate_fn)

loader = {'train' : train_dataloader, 'eval': eval_dataloader ,'test': test_dataloader}

class BerTII(nn.Module):
    def __init__(self, p):
        # p is the Embedding dimension, we get to choose it
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, p)
        self.ln = nn.LayerNorm(p)
        self.linear = nn.Linear(p, 1)
    
    def forward(self, X, N):
        # X.shape : (B, context)
        X = self.embedding_table(X) # (B, context, p)
        X = torch.tensor(torch.nested.to_padded_tensor(X, 0.0), dtype = torch.float)
        X = torch.sum(X, dim = 1) # (B, p)
        X = X / N.unsqueeze(1) # divide each row (prompt) by its length
        X = self.linear(self.ln(X)) # (B, 1)
        logits = torch.sigmoid(X) # (B, 1)
        B = logits.shape[0]
        return logits.view(B) # (B, )

model = BerTII(p).to(device)

# Loss
loss_fn = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

@torch.no_grad
def evaluate_loss():
    out = {}
    model.eval()
    for split in ['eval', 'test']:
        data_loader = loader[split]
        losses = []
        for X, Y, N in data_loader:
            X, Y, N = X.to(device), Y.to(device), N.to(device)
            logits = model(X, N)
            loss = loss_fn(logits, Y)
            losses.append(loss.item())
        out[split] = torch.tensor(losses, dtype = torch.float).mean().item()
    model.train()
    return out

@torch.no_grad
def evaluate_accuracy():
    out = {}
    model.eval()
    for split in ['eval', 'test']:
        data_loader = loader[split]
        acc = 0
        for X, Y, N in data_loader:
            X, Y, N = X.to(device), Y.to(device), N.to(device)
            logits = model(X, N)
            predictions = (logits > 0.5).float()

            # Compare predictions to true targets
            acc += (predictions == Y).sum().item()
        n = len(train_data) # test and train have same length
        acc = acc / n
        out[split] = acc
    model.train()
    return out

# Training Loop
train_iter = iter(train_dataloader)

# Reporing results in a csv file
results = pd.DataFrame(columns= ['Step', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
eval_filename = f'./results-data/sentiment-training_B_{batch_size}_p_{p}.csv'

model.train()
for i in tqdm(range(max_iters)):

    if i % eval_interval == 0:
        losses = evaluate_loss()
        accs = evaluate_accuracy()
        print("Step ", i)
        print(f"Train: Loss : {losses['eval']:.4f}  Accuracy {accs['eval']:.4f} ")
        print(f"Test: Loss {losses['test']:.4f} Accuracy {accs['test']:.4f} ")
        new_row = {"Step": i,
                    'Train Loss': round(losses['eval'], 4),
                    'Train Accuracy': round(accs['eval'] * 100, 2),
                    'Test Loss': round(losses['test'], 4),
                    'Test Accuracy': round(accs['test']*100, 2)
                    }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        results.set_index(['Step']).to_csv(eval_filename)
    
    # sample a batch
    batch = next(train_iter, None)
    if batch is None:
        train_iter = iter(train_dataloader)
        batch = next(train_iter, None)

    X, Y, N = batch
    X, Y, N = X.to(device), Y.to(device), N.to(device)
     
    # Evaluate the loss
    logits = model(X, N)
    optimizer.zero_grad()
    loss = loss_fn(logits, Y)
    loss.backward()
    optimizer.step()

print("End of Training.")
'''
train_iter = iter(train_dataloader)
X, Y, N = next(train_iter)
X, Y, N = X.to(device), Y.to(device), N.to(device)
logits = model(X, N)
loss = loss_fn(logits, Y)
print(loss.item())
#print("Accuracy", evaluate_accuracy())
'''
model_name = f'sentiment_model_B_{batch_size}_p_{p}.pth'
torch.save(model.state_dict(), model_name)
print('Model saved at ', model_name)
