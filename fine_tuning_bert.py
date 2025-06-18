import os
from datasets import load_dataset
import pandas as pd
import torch
from processing.dataset_utils import download_dataset, load_dataset_into_to_dataframe, partition_dataset, IMDBDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models import replace_linear_with_lora
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
import argparse

wandb.login(key='7c2c719a4d241a91163207b8ae5eb635bc0302a4')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)

# Datasets
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_text(batch, truncation = True):
    return tokenizer(batch["text"], truncation=truncation, padding=True)

imdb_dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join("data/sentiment", "train.csv"),
        "validation": os.path.join("data/sentiment", "val.csv"),
        "test": os.path.join("data/sentiment", "test.csv"),
    },
)

imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
del imdb_dataset
imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DataLoaders
train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=os.cpu_count() // 2
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    num_workers=os.cpu_count() // 2
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    num_workers=os.cpu_count() // 2
)
loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}

@torch.no_grad
def evaluate_model(model):
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

def train(model, args):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    n = len(train_loader)
    
    for epoch in range(args.n_epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0

        # Use tqdm for a progress bar
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epochs}")):
            if i > 0 and i % args.inter_eval == 0: # Check i > 0 to avoid eval at step 0
                evals = evaluate_model(model)
                wandb.log({"Val Accuracy": evals["val_acc"], "Test Accuracy": evals["test_acc"]}, step=i)
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
            if i % 10 == 0:
                 wandb.log({"Train Loss (batch)": loss.item()})
        
        # Calculate and log epoch-level metrics
        avg_train_loss = total_train_loss / n
        train_accuracy = total_train_correct / len(train_loader.dataset)
        
        wandb.log({
            "Train Accuracy": train_accuracy,
            "Train Loss (epoch)": avg_train_loss,
            "Epoch": epoch
        })
        print(f'Finished Epoch {epoch+1} / {args.n_epochs}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DistilBERT model with LoRA on IMDB dataset")

    # Training arguments
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--inter_eval", type=int, default=100, help="Steps between intermediate evaluations")

    # LoRA parameters
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=0.5, help="LoRA alpha (input scaling)")
    parser.add_argument("--alpha_r", type=float, default=None, help="LoRA output scaling (defaults to rank)")

    args = parser.parse_args()
    if args.alpha_r is None:
        args.alpha_r = args.rank
    
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = model.to(device)

    # Apply LoRA
    replace_linear_with_lora(model, args.rank, args.alpha, args.alpha_r, device)

    # Print param counts
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The total number of parameters of the BERT model is : {total_params}")
    print(f"The number of trainable parameters after applying LoRA : {lora_params}")

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"BERT-Fine-tuning",

        # track hyperparameters and run metadata
        config={
        "architecture": "DistilBERT",
        "dataset": "IMDB",
        "Alpha": round(args.alpha, 3)
        },
        name = f'alpha_{round(args.alpha, 3)}'
    )
    
    # Start training
    train(model, args)
    print("End of Training.")

    model_name = f'bert_sentiment_model_alpha_{args.alpha}.pth'
    torch.save(model.state_dict(), model_name)
    print('Model saved at ', model_name)

    # Finish the W&B run
    wandb.finish()

    print("End of Training.")
