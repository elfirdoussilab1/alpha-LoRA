# This file is used to combine files
import scipy.io as sio
import numpy as np

# Load the two .mat files
data1 = sio.loadmat('safety_positive.mat')
print(data1['embeddings'].shape)
data2 = sio.loadmat('safety_negative.mat')
print(data2['embeddings'].shape)

# Combine the data (assuming both files have variables you want to merge)
embeddings = np.vstack((data1['embeddings'], data2['embeddings']))
labels = np.hstack((data1['labels'], data2['labels'])).astype(int)[0]
print(labels.shape)
# Random shuffle
l = np.arange(len(embeddings))
np.random.shuffle(l)
embeddings = embeddings[l]
labels = labels[l]

print('combined embeddings shape', embeddings.shape)
print('combined labels shape', labels.shape)

# Add variables from the first file
combined_data = {
    'embeddings': embeddings,
    'labels': labels
}

# Save the dataset to a .mat file
filename = 'safety.mat'
sio.savemat(filename, combined_data)

print(f"Dataset saved successfully to {filename}.")

