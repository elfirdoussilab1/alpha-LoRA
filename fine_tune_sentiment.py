import torch
from dataset import *
from transformers import AutoModelForSequenceClassification, get_scheduler
from model import *
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
import argparse
from utils import fix_seed, evaluate_bert_accuracy

wandb.login(key='7c2c719a4d241a91163207b8ae5eb635bc0302a4')

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a DistilBERT model with Adapters on IMDB dataset")

    # Training arguments
    parser.add_argument("--model_name", type=str, default="roberta-base", help="The model to fine-tune")
    parser.add_argument("--N", type=int, default=None, help="The number of training samples")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lora", type=lambda x: x.lower() == 'true', default=None, help="Boolean to choose if we apply LoRA version")
    parser.add_argument("--lr_adapter", type=float, default=1e-4, help="Learning rate for the Adapter")
    parser.add_argument("--lr_alpha", type=float, default=5e-3, help="Learning rate for alpha")
    parser.add_argument("--inter_eval", type=int, default=200, help="Steps between intermediate evaluations")
    parser.add_argument("--seed", type=int, default=123, help="Random Seed")

    # LoRA parameters
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=None, help="LoRA alpha initialization")
    parser.add_argument("--train_alpha", type=lambda x: x.lower() == 'true', default=None, help="Make alpha trainable or not (True/False)")
    parser.add_argument("--alpha_r", type=float, default=None, help="LoRA output scaling (defaults to rank)")

    args = parser.parse_args()

    # Post-process defaults
    if args.alpha_r is None:
        args.alpha_r = args.rank

    if args.alpha is None:
        args.alpha = np.random.randn()

    return args

def train(model, loader, args):
    adapter_params, alpha_params = optimize_adapter(model)
    param_groups = [{'params': adapter_params, 'lr': args.lr_adapter},
    {'params': alpha_params, 'lr': args.lr_alpha}]

    optimizer = AdamW(param_groups, betas = (0.9, 0.99))
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
                new_alpha = get_alpha(model, args.model_name)
                wandb.log({"Val Accuracy": val_acc, "Test Accuracy": test_acc, "Alpha": new_alpha}, step=epoch * n + i)
                if test_acc > best_acc:
                    best_acc = test_acc
                    print("Saving new best model weights...")
                    path = f'./models/{args.model_name}_sentiment_train_alpha_{args.train_alpha}_lora_{args.lora}.pth'
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
    #apply_adapter(model, args.model_name, args.lora, args.rank, args.alpha, args.alpha_r, device, train_alpha = args.train_alpha)
    replace_linear_with_adapter(model, args.lora, args.rank, args.alpha, args.alpha_r, device, args.train_alpha)
    
    # Print param counts
    total_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The total number of parameters of the model is : {total_params}")
    print(f"The number of trainable parameters after applying LoRA : {adapter_params}")

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"Fine-tuning-Sentiment-{args.model_name}-N-{args.N}",

        # track hyperparameters and run metadata
        config={
        "architecture": args.model_name,
        "dataset": "IMDB",
        "config": vars(args)
        },
        name = f'alpha_trainable_{args.train_alpha}_lora_{args.lora}_init_{round(args.alpha, 3)}'
    )
    
    # Start training
    train(model, loader, args)
    print("End of Training.")

    # Finish the W&B run
    wandb.finish()
