# This ile is used to generate datsets of embeddings

from datasets import load_dataset
from dataset import *
import torch
import numpy as np
import scipy.io as sio
from tqdm.auto import tqdm

dataset_id = 'imdb'
dataset = load_dataset(dataset_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# train data
test_data = dataset['test']

# Prompts and labels
prompts = test_data['text']
labels = test_data['label']

# Generate embeddings for all prompts
embeddings = []
for prompt in tqdm(prompts):
    try:
        embeddings.append(get_embedding_gpt2(prompt, device))
    except:
        continue
    
# Convert labels to a numpy array
labels_np = 2*np.array(labels) - 1

# Prepare data dictionary for saving
data_dict = {
    'embeddings': np.array(embeddings),
    'labels': labels_np
}

# Save the dataset to a .mat file
sio.savemat('sentiment_test_dataset.mat', data_dict)

print("Dataset saved successfully to 'sentiment_test_dataset.mat'.")
