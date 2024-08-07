# This file contains functions to load and preprocess our datasets
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
from bertii_model import *
import json
import csv
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset
import pandas as pd
import torchvision
from torchvision.transforms import ToTensor
from embedders import *

type_to_path = {
    'book' : './datasets/Amazon_review/books.mat',
    'dvd' : './datasets/Amazon_review/dvd.mat',
    'elec' : './datasets/Amazon_review/elec.mat',
    'kitchen' : './datasets/Amazon_review/kitchen.mat',
    'sentiment_train': 'sentiment_train.mat',
    'sentiment_test': 'sentiment_test.mat',
    'sentiment': 'sentiment.mat',
    'safety': 'safety.mat'

}

# Amazon review dataset
class Amazon:
    def __init__(self, n, type_name, classifier = 'pre'):
        # classifier can be either pre-trained (pre) or fine-tuned (ft)
        # Load the dataset
        data = loadmat(type_to_path[type_name])
        self.X = data['fts'] # shape (n, p)

        # Labels
        self.y = data['labels'].reshape((len(self.X), )).astype(int) # shape (n, )
        self.y = 1 - 2 * self.y

        # Preprocessing
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        vmu_1 = np.mean(self.X[self.y < 0], axis = 0)
        vmu_2 = np.mean(self.X[self.y > 0], axis = 0)
        self.mu = np.sqrt(abs(np.inner(vmu_1, vmu_2)))
        self.vmu = (vmu_2 - vmu_1) / 2

        if 'ft' in classifier:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = n / len(self.y)) 
        else: # pre
            self.X_train, self.y_train = self.X[:n], self.y[:n]


# MNIST experiments: Transfer between MNIST classes
class MNIST:
    def __init__(self, n, cl_1, cl_2, classifier = 'pre'):
        # classifier can be either pre-trained (pre) or fine-tuned (ft)
        # cl is a int
        # Load the dataset
        train_data = torchvision.datasets.MNIST(root = "datasets", train = True, download = True, transform = ToTensor())
        test_data = torchvision.datasets.MNIST(root = "datasets", train = False, download = True, transform = ToTensor())

        # Train data
        X_train = train_data.data.cpu().detach().numpy() # shape (n, 28, 28)
        X_train = X_train.reshape(X_train.shape[0], -1) # shape (n, 784)
        y_train = train_data.targets.cpu().detach().numpy() # shape (n, )

        # Test data
        X_test = test_data.data.cpu().detach().numpy() # shape (n, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], -1) # shape (n, 784)
        y_test = test_data.targets.cpu().detach().numpy() # shape (n, )
        
        # Merge
        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))

        # Assign the desired class
        X_1 = X[y == cl_1]
        X_2 = X[y == cl_2]
        y_1 = - np.ones(len(X_1))
        y_2 = np.ones(len(X_2))

        # Get the Binary data
        self.X = np.vstack((X_1, X_2)).astype(float)
        self.y = np.hstack((y_1, y_2)).astype(int)

        # Make the values ranging from -1 to 1
        self.X = (self.X - 127.5) / 127.5

        # Multiply each vector with a normal weight
        #self.X = self.X * np.random.randn(self.X.shape[0], self.X.shape[1]) * 1e-1

        # Add normal vector Z
        self.X = self.X + np.random.randn(self.X.shape[0], self.X.shape[1]) * 1e-1

        # Preprocessing
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        vmu_1 = np.mean(self.X[self.y < 0], axis = 0)
        vmu_2 = np.mean(self.X[self.y > 0], axis = 0)
        self.mu = np.sqrt(abs(np.inner(vmu_1, vmu_2)))
        self.vmu = (vmu_2 - vmu_1) / 2

        if 'ft' in classifier:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = n / len(self.y)) 
        else: # pre
            self.X_train, self.y_train = self.X[:n], self.y[:n]

class MNIST_NN:
    def __init__(self, n, cl_1, cl_2, classifier = 'pre'):
        # classifier can be either pre-trained (pre) or fine-tuned (ft)
        # cl is a int
        # Load the datasets
        train_data = torchvision.datasets.MNIST(root = "datasets", train = True, download = True, transform = ToTensor())
        test_data = torchvision.datasets.MNIST(root = "datasets", train = False, download = True, transform = ToTensor())

        # Train data
        X_train = train_data.data.cpu().detach().numpy() # shape (n, 28, 28)
        X_train = X_train.reshape(X_train.shape[0], -1) # shape (n, 784)
        y_train = train_data.targets.cpu().detach().numpy() # shape (n, )

        # Test data
        X_test = test_data.data.cpu().detach().numpy() # shape (n, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], -1) # shape (n, 784)
        y_test = test_data.targets.cpu().detach().numpy() # shape (n, )
        
        # Merge
        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))

        # Assign the desired class
        X_1 = X[y == cl_1]
        X_2 = X[y == cl_2]
        y_1 = - np.ones(len(X_1))
        y_2 = np.ones(len(X_2))

        # Get the Binary data
        X = np.vstack((X_1, X_2)).astype(float)
        self.y = np.hstack((y_1, y_2)).astype(int)

        # Get the embeddings
        p = 1024
        path = 'mnist_model_6-9-p-1024-B-32.pth'
        embedder = MNISTEmbedder(p, path, 'cpu')
        self.X = embedder.get_embeddings(torch.tensor(X, dtype = torch.float))

        # Preprocessing
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        vmu_1 = np.mean(self.X[self.y < 0], axis = 0)
        vmu_2 = np.mean(self.X[self.y > 0], axis = 0)
        self.mu = np.sqrt(abs(np.inner(vmu_1, vmu_2)))
        self.vmu = (vmu_2 - vmu_1) / 2

        if 'ft' in classifier:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = n / len(self.y)) 
        else: # pre
            self.X_train, self.y_train = self.X[:n], self.y[:n]

def create_pre_ft_datasets(N, type_1, n, type_2, dataset_name):
    if 'amazon' in dataset_name:
        # Pre-training dataset
        data_pre = Amazon(N, type_1, 'pre')

        # Fine-tuning dataset
        data_ft = Amazon(n, type_2, 'ft')

    elif 'mnist' in dataset_name:
        cls_pre = type_1.split('_')
        cls_ft = type_2.split('_')
        # Pre-training dataset
        #data_pre = MNIST(N, int(cls_pre[0]), int(cls_pre[1]), 'pre')
        data_pre = MNIST_NN(N, int(cls_pre[0]), int(cls_pre[1]), 'pre')

        # Fine-tuning dataset
        #data_ft = MNIST(n, int(cls_ft[0]), int(cls_ft[1]), 'ft')
        data_ft = MNIST_NN(n, int(cls_ft[0]), int(cls_ft[1]), 'ft')

    else: # llm
        # Pre-training dataset
        data_pre = LLM_dataset(N, type_1, 'pre')

        # Fine-tuning dataset
        data_ft = LLM_dataset(n, type_2, 'ft')

    # determining beta
    beta = np.inner(data_pre.vmu, data_ft.vmu) / (data_pre.mu**2)

    # Determining orthogonal mu
    if beta < 1:
        vmu_orth = (data_ft.vmu - beta * data_pre.vmu) / np.sqrt(1 - beta**2)
    else:
        vmu_orth = np.zeros_like(data_ft.vmu)
        print(f'Beta {beta} is highe than 1 !')
    return data_pre, data_ft, beta, vmu_orth


# IMDB dataset for sentiment analysis
class Sentiment(Dataset):
    def __init__(self, split, tokenizer, device):
        # split is either train or test
        self.tokenizer = tokenizer
        self.device = device
        #self.context_length = context_length
        ds = load_dataset("stanfordnlp/imdb")
        data = ds[split]
        self.prompts = data['text'] # list of str
        self.labels = torch.tensor(data['label'], dtype = torch.float) # tensor containing 0, 1

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # Tokenization
        tokens = self.tokenizer.encode(self.prompts[idx])
        n = torch.tensor(len(tokens), dtype = torch.float)
        x = torch.tensor(tokens, dtype= torch.long)
        y = self.labels[idx]
        return x.to(self.device), y.to(self.device), n.to(self.device)

class Safety(Dataset):
    def __init__(self, split, tokenizer, device):
        self.device = device
        self.tokenizer = tokenizer
        df = pd.read_csv(f'safety_{split}.csv')
        self.prompts = df['prompt'].tolist()
        self.labels = torch.tensor(df['label'].tolist(), dtype = torch.float)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        # Tokenization
        tokens = self.tokenizer.encode(self.prompts[idx])
        n = torch.tensor(len(tokens), dtype = torch.float)
        x = torch.tensor(tokens, dtype= torch.long)
        y = self.labels[idx]
        return x.to(self.device), y.to(self.device), n.to(self.device)
    
# Collate function: that adds rows of different lengths to the same tensor
def collate_fn(batch):
    X = torch.nested.nested_tensor([x for (x, y, n) in batch])
    X.requires_grad = False
    Y = torch.stack([y for (x, y, n) in batch])
    Y.requires_grad = False
    N = torch.stack([n for (x, y, n) in batch]).view(len(batch))
    N.requires_grad = False
    return X, Y, N

class LLM_dataset:
    def __init__(self, n, type_name, classifier = 'pre') -> None:
        # type_name (str): either 'sentiment' or 'safety'
        data = loadmat(type_to_path[type_name])
        self.X = data['embeddings'] # shape (n, p)
        self.y = data['labels'][0].astype(int)
        assert len(self.y) == self.X.shape[0]
        '''
        # Outlier removal
        gmm = GaussianMixture(n_components=2)
        gmm.fit(self.X)
        # Calculate log-density of each point
        log_density = gmm.score_samples(self.X)
        # Determine threshold for outlier detection
        threshold = np.percentile(log_density, 5)  # Example: 5th percentile
        # Identify outliers
        #outliers = self.X[log_density < threshold]
        # Remove outliers from original dataset
        self.X = self.X[log_density >= threshold]
        self.y = self.y[log_density >= threshold]
        print("Finished outlier removal")
        '''
        # Preprocessing: maybe we should modify this a little 
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        vmu_1 = np.mean(self.X[self.y < 0], axis = 0)
        vmu_2 = np.mean(self.X[self.y > 0], axis = 0)
        self.mu = np.sqrt(abs(np.inner(vmu_1, vmu_2)))
        self.vmu = (vmu_2 - vmu_1) / 2

        if 'ft' in classifier:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = n / len(self.y)) 
        else: # pre
            self.X_train, self.y_train = self.X, self.y


# Get safety dataset
def create_safety_dataset(path = 'unsafe_prompts.jsonl'):# ouputs a csv file
    # Safe prompts
    ds = load_dataset("HuggingFaceH4/ultrachat_200k")
    train_data = ds['train_sft']
    safe_prompts_train = train_data['prompt'][:25000]
    train_labels = [1]*len(safe_prompts_train)

    # Unsafe prompts
    unsafe_prompts = []
    # Open and read the .jsonl file
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())  # Parse each line as JSON
            if 'prompt' in json_obj:  # Check if 'prompt' key exists
                unsafe_prompts.append(json_obj['prompt'])
    n = len(unsafe_prompts)
    unsafe_prompts_train = unsafe_prompts[:n//2]
    train_labels = train_labels + [0]*len(unsafe_prompts_train)
    train_prompts = safe_prompts_train + unsafe_prompts_train

    # Test
    test_data = ds['test_sft']
    safe_prompts_test = test_data['prompt'][:25000]
    test_labels = [1]*len(safe_prompts_test)
    unsafe_prompts_test = unsafe_prompts[n//2:]
    test_prompts = safe_prompts_test + unsafe_prompts_test
    test_labels = test_labels + [0]*len(unsafe_prompts_test)

    print(f'Training data lengths: prompt {len(train_prompts)} label {len(train_labels)}')
    print(f'Test data lengths: prompt {len(test_prompts)} label {len(test_labels)}')
    # Train dataset file
    # File path where you want to save the CSV
    csv_file = 'safety_train.csv'

    # Combine prompts and labels into rows for CSV
    rows = zip(train_prompts, train_labels)

    # Write to CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['prompt', 'label'])  # Write header
        writer.writerows(rows)  # Write rows of prompts and labels

     # Train dataset file
    # File path where you want to save the CSV
    csv_file = 'safety_test.csv'

    # Combine prompts and labels into rows for CSV
    rows = zip(test_prompts, test_labels)

    # Write to CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['prompt', 'label'])  # Write header
        writer.writerows(rows)  # Write rows of prompts and labels

def create_safety_dataset_modified(n, split, path = 'unsafe_prompts.jsonl'):# ouputs a csv file
    # Safe prompts
    ds = load_dataset("HuggingFaceH4/ultrachat_200k")
    data = ds[f'{split}_sft']
    safe_prompts = data['prompt'][:3*n//5]
    labels = [1]*len(safe_prompts)

    # Unsafe prompts
    unsafe_prompts = []
    # Open and read the .jsonl file
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())  # Parse each line as JSON
            if 'prompt' in json_obj:  # Check if 'prompt' key exists
                unsafe_prompts.append(json_obj['prompt'])

    unsafe_prompts = unsafe_prompts[:2*n//5]
    labels = labels + [0]*len(unsafe_prompts)
    prompts = safe_prompts + unsafe_prompts

    print(f'{split} data lengths: prompts {len(prompts)} label {len(labels)}')
    # Train dataset file
    # File path where you want to save the CSV
    csv_file = f'safety_{split}_n_{n}.csv'

    # Combine prompts and labels into rows for CSV
    rows = zip(prompts, labels)

    # Write to CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['prompt', 'label'])  # Write header
        writer.writerows(rows)  # Write rows of prompts and labels

#create_safety_dataset_modified(n = 1000, split = 'train')

# MNIST dataset to get embeddings
# Dataset
class CustomMnistDataset(Dataset):
    def __init__(self, cl_1, cl_2, train, device = 'cpu'):
        super().__init__()
        self.device = device

        data = torchvision.datasets.MNIST(root = "datasets", train = train, download = True, transform = ToTensor())
        X = data.data.cpu().detach().numpy()
        X = X.reshape(X.shape[0], -1)
        y = data.targets.cpu().detach().numpy()
        X_1 = X[y == cl_1]
        X_2 = X[y == cl_2]
        y_1 = np.zeros(len(X_1))
        y_2 = np.ones(len(X_2))
        X = np.vstack((X_1, X_2)).astype(float)
        y = np.hstack((y_1, y_2)).astype(int)
        
        # Converting back to tensor
        self.X = torch.tensor(X, dtype = torch.float)
        self.labels = torch.tensor(y, dtype= torch.float)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Tokenization
        x = self.X[idx]
        y = self.labels[idx]
        return x.to(self.device), y.to(self.device)