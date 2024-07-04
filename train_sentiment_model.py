# This file will be used to train the sentiment analysis model
from sentiment_model import *
import tiktoken
from torch.utils.data import DataLoader
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
p = 50
#----------

# For reproducibility
torch.manual_seed(1337)

# Tokenizer
tokenizer = tiktoken.get_encoding("o200k_base")
vocab_size = tokenizer.max_token_value

# Datasets 
train_data = Sentiment('train', tokenizer, device)
test_data = Sentiment('test', tokenizer, device)

# Collate function: that adds rows of different lengths to the same tensor
def collate_fn(batch):
    X = torch.nested.nested_tensor([x for (x, y, n) in batch])
    X.requires_grad = False
    Y = torch.stack([y for (x, y, n) in batch])
    Y.requires_grad = False
    N = torch.stack([n for (x, y, n) in batch]).view(len(batch))
    N.requires_grad = False
    return X, Y, N

# DataLoaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)
eval_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn= collate_fn)
test_dataloader = DataLoader(test_data, batch_size= batch_size , shuffle=True, collate_fn= collate_fn)

loader = {'train' : train_dataloader, 'eval': eval_dataloader ,'test': test_dataloader}

# Model
model = BerTII(p, vocab_size).to(device)

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

model_name = f'sentiment_model_B_{batch_size}_p_{p}.pth'
torch.save(model.state_dict(), model_name)
print('Model saved at ', model_name)
