# This file contains functions to load and preprocess our datasets
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Union
from datasets import load_dataset
import torch
from sentiment_model import *
import json
import csv
from sklearn.mixture import GaussianMixture

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
            self.X_train, self.y_train = self.X, self.y


class LLM_dataset:
    def __init__(self, n, type_name, classifier = 'pre') -> None:
        # type_name (str): either 'sentiment' or 'safety'
        data = loadmat(type_to_path[type_name])
        self.X = data['embeddings'] # shape (n, p)
        self.y = data['labels'][0].astype(int)
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

        # Preprocessing: maybe we should modify this a little 
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        print("Finished standard scaling")
        
        vmu_1 = np.mean(self.X[self.y < 0], axis = 0)
        vmu_2 = np.mean(self.X[self.y > 0], axis = 0)
        self.mu = np.sqrt(abs(np.inner(vmu_1, vmu_2)))
        self.vmu = (vmu_2 - vmu_1) / 2

        if 'ft' in classifier:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = n / len(self.y)) 
        else: # pre
            self.X_train, self.y_train = self.X, self.y

def create_pre_ft_datasets(N, type_1, n, type_2, dataset_name = 'amazon'):
    if 'amazon' in dataset_name:
        # Pre-training dataset
        data_pre = Amazon(N, type_1, 'pre')

        # Fine-tuning dataset
        data_ft = Amazon(n, type_2, 'ft')

    else: # llm
        # Pre-training dataset
        data_pre = LLM_dataset(N, type_1, 'pre')

        # Fine-tuning dataset
        data_ft = LLM_dataset(n, type_2, 'ft')

    # determining beta
    beta = np.inner(data_pre.vmu, data_ft.vmu) / (data_pre.mu**2)

    # Determining orthogonal mu
    vmu_orth = (data_ft.vmu - beta * data_pre.vmu) / np.sqrt(1 - beta**2)

    return data_pre, data_ft, beta, vmu_orth

# Get safety dataset
def create_safety_dataset(path = 'unsafe_prompts.jsonl'):
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
    unsafe_prompts_test = unsafe_prompts[n:]
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

    
#create_safety_dataset()
