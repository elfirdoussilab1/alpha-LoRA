import torch
from dataset import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import *
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
import argparse
from utils import fix_seed

wandb.login(key='7c2c719a4d241a91163207b8ae5eb635bc0302a4')

fix_seed(123)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DistilBERT model with LoRA on IMDB dataset")

    # Training arguments
    parser.add_argument("--model_name", type=str, default="roberta-base", help="The model to fine-tune")
    parser.add_argument("--N", type=int, default=None, help="The number of training samples")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--inter_eval", type=int, default=200, help="Steps between intermediate evaluations")

    # LoRA parameters
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=1, help="LoRA alpha (input scaling)")
    parser.add_argument("--alpha_r", type=float, default=None, help="LoRA output scaling (defaults to rank)")

    args = parser.parse_args()
    if args.alpha_r is None:
        args.alpha_r = args.rank
    
    return args

@torch.no_grad
def evaluate_model(model, loader):
    out = {}
    model.eval()

    for split in ['val', 'test']:
        data_loader = loader[split]
        total_correct = 0
        total_samples = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Perform a forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Calculate accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0) # Add the number of samples in the current batch
        
        # Calculate accuracy over the entire dataset
        out[split + '_acc'] = total_correct / total_samples
    return out

def train(model, loader, args):
    optimizer = AdamW(model.parameters(), lr=args.lr, betas = (0.9, 0.99))
    n = len(loader['train'])
    best_acc = 0
    for epoch in range(args.n_epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0

        # Use tqdm for a progress bar
        for i, batch in enumerate(tqdm(loader['train'], desc=f"Epoch {epoch+1}/{args.n_epochs}")):
            if i % args.inter_eval == 0 or i == n: 
                evals = evaluate_model(model, loader)
                wandb.log({"Val Accuracy": evals["val_acc"], "Test Accuracy": evals["test_acc"]}, step= epoch * n + i)
                if evals["test_acc"] > best_acc:
                    best_acc = evals["test_acc"]
                    print("Saving new best model weights...")
                    path = f'./models/{args.model_name}_sentiment_alpha_{args.alpha}.pth'
                    torch.save(model.state_dict(), path)
                    print('Model saved at: ', path)
                model.train()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_train_correct += (predictions == labels).sum().item()
            
            # Log batch loss less frequently or not at all, to avoid noisy charts
            if i % 20 == 0:
                 wandb.log({"Train Loss (batch)": loss.item()})
        
        # Calculate and log epoch-level metrics
        avg_train_loss = total_train_loss / n
        train_accuracy = total_train_correct / len(loader['train'].dataset)
        
        wandb.log({
            "Train Accuracy": train_accuracy,
            "Train Loss (epoch)": avg_train_loss
        })
        print(f'Finished Epoch {epoch+1} / {args.n_epochs}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}')

if __name__ == "__main__":
    
    args = parse_args()
    # Tokenizer and datasets
    imdb_tokenized = tokenization(args.model_name)

    # DataLoaders
    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    if args.N is not None:
        train_dataset = sample_n(train_dataset, args.N)

    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        num_workers=2
    )
    loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model = model.to(device)

    # Apply LoRA
    apply_lora(model, args.model_name, args.rank, args.alpha, args.alpha_r, device, train_alpha= False)
    
    # Print param counts
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The total number of parameters of the BERT model is : {total_params}")
    print(f"The number of trainable parameters after applying LoRA : {lora_params}")

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"Fine-tuning-{args.model_name}-N-{args.N}",

        # track hyperparameters and run metadata
        config={
        "architecture": args.model_name,
        "dataset": "IMDB",
        "Alpha": round(args.alpha, 3)
        },
        name = f'alpha_{round(args.alpha, 3)}'
    )
    
    # Start training
    train(model, args)
    print("End of Training.")

    # Finish the W&B run
    wandb.finish()
