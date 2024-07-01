# This file is used to generate safety embeddings dataset

from dataset import *
import torch
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

embedder = GPT2Embedder(device=device)

# Load dataset: Ultra_chat
ds = load_dataset("HuggingFaceH4/ultrachat_200k")
train_data = ds['train_sft']
#test_data = ds['test_sft']

embeddings = embedder.get_embeddings(train_data, batch_size= 20)
print(embeddings.shape)
labels = np.ones(len(embeddings))

# Prepare data dictionary for saving
data_dict = {
    'embeddings': np.array(embeddings),
    'labels': labels
}

# Save the dataset to a .mat file
filename = 'safety_train.mat'
sio.savemat(filename, data_dict)

print(f"Dataset saved successfully to {filename}.")


