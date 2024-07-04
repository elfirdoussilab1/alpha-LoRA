# This file is used to generate safety embeddings dataset
from datasets import load_dataset
import torch
import numpy as np
import scipy.io as sio
from embedders import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
p = 1000
path = f'sentiment_model_B_64_p_{p}.pth'
embedder = CustomEmbdedder(p, path, device)

# Load dataset: Ultra_chat
ds = load_dataset("HuggingFaceH4/ultrachat_200k")
#data = ds['train_sft']
data = ds['test_sft']
prompts = data['prompt'][:25000]

embeddings = embedder.get_embeddings(prompts, batch_size= 100)
print('Embeddings shape', embeddings.shape)
labels = np.ones(len(embeddings))

# Prepare data dictionary for saving
data_dict = {
    'embeddings': np.array(embeddings),
    'labels': labels
}

# Save the dataset to a .mat file
filename = 'safety_positive_test.mat'
sio.savemat(filename, data_dict)

print(f"Dataset saved successfully to {filename}.")
