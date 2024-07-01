# This file is used to combine files
import scipy.io as sio
import numpy as np

# Load the two .mat files
data1 = sio.loadmat('sentiment_train_dataset_1.mat')
print(data1['embeddings'].shape)
data2 = sio.loadmat('sentiment_train_dataset_2.mat')
print(data2['embeddings'].shape)

# Combine the data (assuming both files have variables you want to merge)
embeddings = np.vstack((data1['embeddings'].reshape(-1, 768), data2['embeddings'].reshape(-1, 768)))
print('combined embeddings shape', embeddings.shape)
labels = np.hstack((data1['labels'], data2['labels'])).astype(int)
print('combined labels shape', labels.shape)

# Add variables from the first file
combined_data = {
    'embeddings': embeddings,
    'labels': labels
}

# Save the dataset to a .mat file
filename = 'sentiment_train.mat'
sio.savemat(filename, combined_data)

print(f"Dataset saved successfully to {filename}.")

