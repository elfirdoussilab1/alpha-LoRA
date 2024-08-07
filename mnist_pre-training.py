import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import wandb
from tqdm.auto import tqdm
from models import simple_mnist
from dataset import CustomMnistDataset

wandb.login(key='7c2c719a4d241a91163207b8ae5eb635bc0302a4')

# For reproducibility
torch.manual_seed(1337)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Datasets
# Classes
cl_1 = 6
cl_2 = 9

train_data = CustomMnistDataset(cl_1, cl_2, train = True, device = device)
test_data = CustomMnistDataset(cl_1, cl_2, train = False, device = device)

# Dataloader
batch_size = 32
train_loader = DataLoader(train_data, batch_size= batch_size, shuffle = True)
eval_loader = DataLoader(train_data, batch_size= batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size= batch_size, shuffle = True)

loader = {'train' : train_loader, 'eval': eval_loader ,'test': test_loader}

# Hyperparams 
max_iters = 1000
eval_interval = 20
lr = 1e-3

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Pretraining-MNIST-6-9",

    # track hyperparameters and run metadata
    config={
    "architecture": "simple NN",
    "dataset": "MNIST",
    "lr" : lr
    }
)

############## Model, Loss and Optimizer ##############
p = 1024
model = simple_mnist(p).to(device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay= 1e-2)

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
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            predictions = (logits > 0.5).float()

            # Compare predictions to true targets
            acc += (predictions == Y).sum().item()
        if 'eval' in split:
            n = len(train_data) 
        else:
            n = len(test_data) 
        acc = acc / n
        out[split] = acc
    model.train()
    return out

# Training Loop
train_iter = iter(train_loader)

model.train()
for i in tqdm(range(max_iters)):

    if i % eval_interval == 0:
        losses = evaluate_loss()
        accs = evaluate_accuracy()
        print("Step ", i)
        print(f"Train: Loss : {losses['eval']:.4f}  Accuracy {accs['eval']:.4f} ")
        print(f"Test: Loss {losses['test']:.4f} Accuracy {accs['test']:.4f} ")
        
    # log metrics to wandb
    wandb.log({"Train loss": losses['eval'], "Test loss": losses['test'], "Train Accuracy": accs['eval'], "Test Accuracy": accs['test']})

    # sample a batch
    batch = next(train_iter, None)
    if batch is None:
        train_iter = iter(train_loader)
        batch = next(train_iter, None)

    X, Y = batch
    X, Y = X.to(device), Y.to(device)
     
    # Evaluate the loss
    logits = model(X)
    optimizer.zero_grad()
    loss = loss_fn(logits, Y)
    loss.backward()
    optimizer.step()

# Save model
model_name = f'mnist_model_{cl_1}-{cl_2}-p-{p}-B-{batch_size}.pth'
torch.save(model.state_dict(), model_name)

# Finish the W&B run
wandb.finish()

print("End of Training.")