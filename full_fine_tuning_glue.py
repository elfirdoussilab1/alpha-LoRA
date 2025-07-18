# In this file, we implement the finetuning experiments of roberta-base model on GLUE tasks: MNLI, QP and QNLI.
import torch
from dataset import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from model import *
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
import argparse
from utils import fix_seed, evaluate_bert_accuracy

wandb.login(key='7c2c719a4d241a91163207b8ae5eb635bc0302a4')

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DistilBERT model with LoRA on GLUE task")

    # Training arguments
    parser.add_argument("--model_name", type=str, default="roberta-base", help="The model to fine-tune")
    parser.add_argument("--task_name", type=str, default=None, help="The desired dataset")
    parser.add_argument("--N", type=int, default=None, help="The number of training samples")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the whole model")
    parser.add_argument("--inter_eval", type=int, default=200, help="Steps between intermediate evaluations")
    parser.add_argument("--seed", type=int, default=123, help="Random Seed")

    args = parser.parse_args()

    return args

def train(model, loader, args):
    optimizer = AdamW(model.parameters(), lr=args.lr, betas = (0.9, 0.99))
    n = len(loader['train'])
    best_acc = 0
    num_training_steps = args.n_epochs * n
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    for epoch in range(args.n_epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0

        # Use tqdm for a progress bar
        for i, batch in enumerate(tqdm(loader['train'], desc=f"Epoch {epoch+1}/{args.n_epochs}")):
            if i % args.inter_eval == 0 or i == -1: 
                test_acc = evaluate_bert_accuracy(model, loader['test'], device)
                val_acc = evaluate_bert_accuracy(model, loader['val'], device)
                wandb.log({"Val Accuracy": val_acc, "Test Accuracy": test_acc}, step=epoch * n + i)
                if test_acc > best_acc:
                    best_acc = test_acc
                    print("Saving new best model weights...")
                    path = f'./models/{args.model_name}_{args.task_name}_full.pth'
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
            lr_scheduler.step()

            total_train_loss += loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_train_correct += (predictions == labels).sum().item()
            
            # Log batch loss less frequently or not at all, to avoid noisy charts
            if i % 10 == 0:
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device)
    args = parse_args()
    fix_seed(args.seed)
    
    # Tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_data, val_data, test_data = get_glue_datasets(args.task_name)
    # Define the sentence keys for each GLUE task. Most have two sentences.
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[args.task_name.lower()]
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key],padding="max_length",truncation=True, max_length=128)
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_val = val_data.map(preprocess_function, batched=True)
    tokenized_test = test_data.map(preprocess_function, batched=True)

    # Remove original text columns and set format to PyTorch tensors
    tokenized_train = tokenized_train.remove_columns([k for k in task_to_keys[args.task_name.lower()] if k is not None] + ['idx'])
    tokenized_val = tokenized_val.remove_columns([k for k in task_to_keys[args.task_name.lower()] if k is not None] + ['idx'])
    tokenized_test = tokenized_test.remove_columns([k for k in task_to_keys[args.task_name.lower()] if k is not None] + ['idx'])
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    tokenized_test.set_format("torch")

    # Create DataLoaders
    train_loader = DataLoader(tokenized_train, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(tokenized_val, batch_size=args.batch_size)
    test_loader = DataLoader(tokenized_test, batch_size=args.batch_size)
    loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # Model
    num_labels = train_data.features['label'].num_classes
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels
    )
    print("Number of classes: ", num_labels)
    model = model.to(device)

    # Make all parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    
    # Print param counts
    total_params = sum(p.numel() for p in model.parameters())
    tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The total number of parameters of the model is : {total_params}")
    print(f"The number of trainable parameters : {tr_params}")

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"Fine-tuning-{args.task_name.upper()}-{args.model_name}-N-{args.N}",

        # track hyperparameters and run metadata
        config={
        "architecture": args.model_name,
        "dataset": args.task_name.upper(),
        "config": vars(args)
        },
        name = f'full_fine_tuning_seed_{args.seed}'
    )
    
    # Start training
    train(model, loader, args)
    print("End of Training.")

    # Finish the W&B run
    wandb.finish()
