# This file aims at implementing a method to estimate beta
# We will first try the method on the sentiment dataset, then create a general 
# function to estimate beta for any dataset
import torch
from transformers import AutoModelForSequenceClassification
from model import *
from dataset import *
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sentence_transformers import SentenceTransformer
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate the beta")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="The model to fine-tune")
    parser.add_argument("--sentence_model", type= str, default="all-MiniLM-L6-v2", help="The embedding model")
    args = parser.parse_args()
    return args

args = parse_args()
# Load the model and tokenizer
model_name = args.model_name #'bert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = model.to(device)
model.eval()

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": op.join("data/sentiment", "train.csv")
        },
)
imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
os.environ["TOKENIZERS_PARALLELISM"] = "false"
train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")

batch_size = 32
dataloader = DataLoader(train_dataset, batch_size=batch_size)

labels_noisy = []

# Compute noisy labels
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
        logits = outputs.logits
        probs = softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        labels_noisy.extend(preds.cpu().tolist())

labels_noisy = np.array(labels_noisy)
print("Computed the predictions:", labels_noisy)
# Find the dominating label
if np.sum(labels_noisy) > len(labels_noisy) // 2:
    y = 1
else:
    y = 0


print("The dominating label is: ", y)
print("Now vectorizing the dataset...")

# Vectorize the dataset
embedder = SentenceTransformer(args.sentence_model)
print("Loaded the sentence transformer")
train_set = imdb_dataset['train']
texts = train_set['text']

embeddings = embedder.encode(
    texts, 
    batch_size=32, 
    show_progress_bar=True,
    convert_to_numpy=True,  # or convert_to_tensor=True if using PyTorch later
    normalize_embeddings=True  # set True if using cosine similarity
    )

print("Shape of embeddings is ", embeddings.shape)
print("Successfully created the embeddings")
print("Computing the means")
# Computing the means
labels_true = np.array(train_set['label']) # list

if y == 0:
    # Change the 0 to 1 and 1 to 0
    labels_noisy = (labels_noisy + 1) % 2
    labels_true = (labels_true + 1) % 2
vmu = np.mean(labels_noisy[:, np.newaxis] * embeddings, axis = 0)
vmu_beta = np.mean(labels_true[:, np.newaxis] * embeddings, axis = 0)

print(f"Shape of vmu and vmu_beta resp: {vmu.shape} and {vmu_beta.shape}")
mu = np.linalg.norm(vmu)
mu_beta = np.linalg.norm(vmu_beta)

beta = np.sum(vmu * vmu_beta) / (mu* mu_beta)
print(f"Beta is given by: {beta}")
