# This file is used to generate datsets of embeddings
from datasets import load_dataset
from dataset import *
import torch
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm

dataset_id = 'stanfordnlp/imdb'
dataset = load_dataset(dataset_id)
p = 1000
path = f'sentiment_model_B_64_p_{p}.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# train data
train_data = dataset['train']

# Prompts and labels
prompts = train_data['text']
labels = train_data['label']

# Generate embeddings for all prompts
embedder = CustomEmbdedder(p, path, device)
embeddings = embedder.get_embeddings(prompts, batch_size= 100)
    
# Convert labels to a numpy array
labels_np = 2*np.array(labels) - 1

# Prepare data dictionary for saving
data_dict = {
    'embeddings': np.array(embeddings),
    'labels': labels_np
}

# Save the dataset to a .mat file
filename = 'sentiment_test_dataset.mat'
sio.savemat(filename, data_dict)

print(f"Dataset saved successfully to {filename}'.")
