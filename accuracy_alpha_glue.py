# In this file we will implement the experiment of evaluting a model on a neighborhood of the obtained optimal alpha
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataset import *
from model import *
from tqdm.auto import tqdm
from copy import deepcopy
from utils import fix_seed, evaluate_bert_accuracy
from torch.utils.data import DataLoader
import argparse

# Small arg parser
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the test accuracy with alpha")

    # Training arguments
    parser.add_argument("--model_name", type=str, default="roberta-base", help="The finetuned model")
    # we can slso use: Qwen/Qwen2.5-0.5B or google/gemma-3-270m or meta-llama/Llama-3.2-1B
    parser.add_argument("--task_name", type=str, default=None, help="The desired dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device)
    args = parse_args()

    # Load the test dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_data, val_data, test_data = get_glue_datasets(args.task_name, 0.1)
    # Define the sentence keys for each GLUE task. Most have two sentences.
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "wnli": ("sentence1", "sentence2")
    }
    sentence1_key, sentence2_key = task_to_keys[args.task_name.lower()]
    def preprocess_function(examples):
        max_length = 512 if args.task_name.lower() in ["mnli", "qnli"] else 128
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key],padding="max_length",truncation=True, max_length=max_length)
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_test = test_data.map(preprocess_function, batched=True)

    # Remove original text columns and set format to PyTorch tensors
    tokenized_test = tokenized_test.remove_columns([k for k in task_to_keys[args.task_name.lower()] if k is not None] + ['idx'])
    tokenized_test.set_format("torch")

    test_loader = DataLoader(tokenized_test, batch_size=args.batch_size)
    num_labels = train_data.features['label'].num_classes

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=num_labels
        ).to(device)
    # Apply LoRA
    apply_adapter(model, args.model_name, lora = True, rank = args.rank, alpha= 1, alpha_r= args.rank, device =device, train_alpha = True)
    model.load_state_dict(torch.load(f'models/{args.model_name}_{args.task_name}_alpha_trainable_True.pth'))
    model.eval()

    # Evaluation part
    A = np.linspace(-2, 2, 50)
    B = np.linspace(-2, 2, 50)
    accs = np.zeros((len(A), len(B)))

    def add_to_alpha(model, a, b):
        new_model = deepcopy(model)
        for name, param in new_model.named_parameters():
            if 'alpha' in name:
                with torch.no_grad():  # avoid tracking in autograd 
                    # in-place multiplication and addition
                    param.mul_(a) 
                    param.add_(b)
        return new_model

    for i, a in enumerate(tqdm(A)):
        for j, b in enumerate(B):
            # clone the model and add the constant to each parameter alpha
            new_model = add_to_alpha(model, a, b)
            new_model.to(device)
            # Evaluate this model
            test_acc = evaluate_bert_accuracy(new_model, test_loader, device)
            accs[i, j] = test_acc
    print(accs)
    # Save the accuracies list to npy object
    np.save("accs_alpha.npy", np.array(accs))