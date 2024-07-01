# In this file, we will create the model that we will use to generate our embeddings
# Our model will be trained on the sentiment analysis task
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Hyperparameters
batch_size = 64
#T = 2000 # Context length
max_iters = 5000
eval_interval = 500
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device', device)
eval_iters = 200

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
        x = torch.tensor(tokens, dtype= torch.long)
        y = self.labels[idx]
        return x.to(device), y.to(device)

# Collate function: that adds rows of different lengths to the same tensor
def collate_fn(batch):
    X = torch.nested.nested_tensor([x for (x, y) in batch])
    Y = torch.stack([y for (x, y) in batch])
    return X, Y

# Tokenizer
tokenizer = tiktoken.get_encoding("o200k_base")
vocab_size = tokenizer.max_token_value

# Datasets 
train_data = Sentiment('train', tokenizer)
test_data = Sentiment('test', tokenizer)

# DataLoaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)
eval_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, collate_fn= collate_fn)
test_dataloader = DataLoader(test_data, batch_size= batch_size , shuffle=True, collate_fn= collate_fn)

loader = {'train' : train_dataloader, 'eval': eval_dataloader ,'test': test_dataloader}

class BerTII(nn.Module):
    def __init__(self, p) -> None:
        # p is the Embedding dimension, we get to choose it
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, p)
        self.linear = nn.Linear(p, 1)
    
    def forward(self, X):
        # X.shape : (B, context)
        X = self.embedding_table(X) # (B, context, p)
        X = torch.nested.to_padded_tensor(X, 0.0)
        X = torch.mean(X, dim = 1) # (B, p)
        X = self.linear(X) # (B, 1)
        logits = torch.sigmoid(X) # (B, 1)
        return logits.view(batch_size) # (B, )

model = BerTII(p)

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
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = loss_fn(logits, Y)
            losses.append(loss.item())
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad
def evaluate_accuracy():
    out = {}
    model.eval()
    for split in ['eval', 'test']:
        data_loader = loader[split]
        acc = 0
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
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

model.train()
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = evaluate_loss()
        accs = evaluate_accuracy()
        print("Step ", iter)
        print(f"Train: Loss : {losses['eval']:.4f}  Accuracy {accs['eval']:.4f} ")
        print(f"Test: Loss {losses['test']:.4f} Accuracy {accs['test']:.4f} ")
    
    # sample a batch
    X, Y = next(train_iter)
    X, Y = X.to(device), Y.to(device)
     
    # Evaluate the loss
    logits = model(X)
    optimizer.zero_grad()
    loss = loss_fn(logits, Y)
    loss.backward()
    optimizer.step()

"""
train_iter = iter(train_dataloader)
X, Y = next(train_iter)
X, Y = X.to(device), Y.to(device)
logits = model(X)
loss = loss_fn(logits, Y)
print(loss.item())
print("Accuracy", evaluate_accuracy())
"""
print("End of Training.")


    











