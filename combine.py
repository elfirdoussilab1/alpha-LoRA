# This file is used to combine files
import scipy.io as sio
import numpy as np

# Load the two .mat files
data1 = sio.loadmat('sentiment_train_dataset.mat')
print(data1['embeddings'].shape)
data2 = sio.loadmat('sentiment_test_dataset.mat')
print(data2['embeddings'].shape)

# Combine the data (assuming both files have variables you want to merge)
embeddings = np.vstack((data1['embeddings'], data2['embeddings']))
print('combined embeddings shape', embeddings.shape)
labels = np.hstack((data1['labels'], data2['labels'])).astype(int)
print('combined labels shape', labels.shape)

# Add variables from the first file
combined_data = {
    'embeddings': embeddings,
    'labels': labels
}

# Save the dataset to a .mat file
filename = 'sentiment.mat'
sio.savemat(filename, combined_data)

print(f"Dataset saved successfully to {filename}.")

