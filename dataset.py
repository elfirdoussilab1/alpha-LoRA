# This file contains functions to load and preprocess our datasets
# Standard library
import os
import os.path as op
import sys
import tarfile
import time
import json
import csv
import urllib
import random

# Third-party libraries
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.transforms import ToTensor
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_california_housing, load_diabetes
from datasets import load_dataset
from packaging import version
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Tuple

# Local modules
from model import *

type_to_path = {
    'book' : './data/Amazon_review/books.mat',
    'dvd' : './data/Amazon_review/dvd.mat',
    'elec' : './data/Amazon_review/elec.mat',
    'kitchen' : './data/Amazon_review/kitchen.mat'
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
        data_pre = MNIST(N, int(cls_pre[0]), int(cls_pre[1]), 'pre')

        # Fine-tuning dataset
        data_ft = MNIST(n, int(cls_ft[0]), int(cls_ft[1]), 'ft')

    else: # llm
        # Pre-training dataset
        data_pre = LLM_dataset(N, type_1, 'pre')

        # Fine-tuning dataset
        data_ft = LLM_dataset(n, type_2, 'ft')

    # determining beta
    beta = np.inner(data_pre.vmu, data_ft.vmu) / (data_pre.mu**2)

    # Determining orthogonal mu
    vmu_orth = data_ft.vmu - beta * data_pre.vmu
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

## IMDB dataset for the new Sentiment Analysis fine-tuning
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.0**2 * duration)
    percent = count * block_size * 100.0 / total_size

    sys.stdout.write(
        f"\r{int(percent)}% | {progress_size / (1024.**2):.2f} MB "
        f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
    )
    sys.stdout.flush()


def download_dataset():
    source = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    target = "aclImdb_v1.tar.gz"

    if os.path.exists(target):
        os.remove(target)

    if not os.path.isdir("aclImdb") and not os.path.isfile("aclImdb_v1.tar.gz"):
        urllib.request.urlretrieve(source, target, reporthook)

    if not os.path.isdir("aclImdb"):

        with tarfile.open(target, "r:gz") as tar:
            tar.extractall()


def load_dataset_into_to_dataframe():
    basepath = "aclImdb"

    labels = {"pos": 1, "neg": 0}

    df = pd.DataFrame()

    with tqdm(total=50000) as pbar:
        for s in ("test", "train"):
            for l in ("pos", "neg"):
                path = os.path.join(basepath, s, l)
                for file in sorted(os.listdir(path)):
                    with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                        txt = infile.read()

                    if version.parse(pd.__version__) >= version.parse("1.3.2"):
                        x = pd.DataFrame(
                            [[txt, labels[l]]], columns=["review", "sentiment"]
                        )
                        df = pd.concat([df, x], ignore_index=False)

                    else:
                        df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()
    df.columns = ["text", "label"]

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))

    print("Class distribution:")
    np.bincount(df["label"].values)

    return df


def partition_dataset(df):
    df_shuffled = df.sample(frac=1, random_state=1).reset_index()

    df_train = df_shuffled.iloc[:35_000]
    df_val = df_shuffled.iloc[35_000:40_000]
    df_test = df_shuffled.iloc[40_000:]

    if not op.exists("data"):
        os.makedirs("data")
    df_train.to_csv(op.join("data/sentiment", "train.csv"), index=False, encoding="utf-8")
    df_val.to_csv(op.join("data/sentiment", "val.csv"), index=False, encoding="utf-8")
    df_test.to_csv(op.join("data/sentiment", "test.csv"), index=False, encoding="utf-8")


class IMDBDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows


def get_dataset():
    files = ("test.csv", "train.csv", "val.csv")
    download = True

    for f in files:
        if not os.path.exists(f):
            download = False

    if download is False:
        download_dataset()
        df = load_dataset_into_to_dataframe()
        partition_dataset(df)

    df_train = pd.read_csv(op.join("data", "train.csv"))
    df_val = pd.read_csv(op.join("data", "val.csv"))
    df_test = pd.read_csv(op.join("data", "test.csv"))

    return df_train, df_val, df_test


def tokenization(model_name):
    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": op.join("data/sentiment", "train.csv"),
            "validation": op.join("data/sentiment", "val.csv"),
            "test": op.join("data/sentiment", "test.csv"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_text(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return imdb_tokenized

# Sample N samples from a dataset
def sample_n(dataset, n):
    assert n <= len(dataset), "Cannot sample more than the dataset size"
    indices = random.sample(range(len(dataset)), n)
    return Subset(dataset, indices)


def get_glue_datasets(task_name: str, val_split_ratio: float = 0.2, seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads the train, validation (subset of train), and test datasets for a specified GLUE task.

    Args:
        task_name (str): The name of the GLUE task (e.g., 'MNLI', 'QQP', 'QNLI').
        val_split_ratio (float): The proportion of the training set to use for validation.
        seed (int): Random seed for reproducibility of the split.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple containing the training,
                                           validation, and testing datasets.
                                           The test set for MNLI will be the 'matched' version.
    """
    task_name_lower = task_name.lower()
    
    # Load the dataset from the Hugging Face Hub
    print(f"🔄 Loading GLUE dataset for task: {task_name_lower}...")
    raw_datasets = load_dataset('glue', task_name_lower)
    print("✅ Dataset loaded successfully.")

    # GLUE's MNLI task has unique test split names
    if task_name_lower == 'mnli':
        test_key = 'validation_mismatched'  # Use mismatched as test
    else:
        test_key = 'validation'  # fallback to val for testing
    
    # Split train into new train and validation sets
    if val_split_ratio > 0:
        if val_split_ratio < 1:
            split_dataset = raw_datasets['train'].train_test_split(test_size=val_split_ratio, seed=seed)
            #train_dataset = split_dataset['train']
            train_dataset = raw_datasets['train']
            validation_dataset = split_dataset['test']
        else:
            train_dataset = raw_datasets['train']
            validation_dataset = raw_datasets['train']

    else:
        train_dataset = raw_datasets['train']
        validation_dataset = raw_datasets[test_key]
    test_dataset = raw_datasets[test_key]

    # Set the format to PyTorch tensors
    train_dataset.set_format('torch')
    validation_dataset.set_format('torch')
    test_dataset.set_format('torch')

    return train_dataset, validation_dataset, test_dataset


##### Linear Regression Datasets #####

class LinearRegressionDatasetLoader:
    def __init__(
        self, dataset_name, n, p = None, d = None, sigma=0.0, whiten=False, random_state=42, project_features=False
    ):
        self.dataset_name = dataset_name.lower()
        self.sigma = sigma
        self.whiten = whiten
        self.n = n
        self.random_state = random_state
        self.p = p
        self.d = d
        self.project_features = project_features

    def get_data(self):
        X, Y = self._load_dataset()
        n_train = len(X)
        # Standardize inputs
        X = StandardScaler().fit_transform(X)

        # Reduce feature dimension if needed
        if self.p is not None:
            pca_X = PCA(n_components=self.p, whiten=self.whiten)
            X = pca_X.fit_transform(X)

        # Standardize outputs
        Y = StandardScaler().fit_transform(Y)

        # Reduce target dimension if needed
        if self.d is not None and Y.shape[1] > self.d:
            pca_Y = PCA(n_components=self.d)
            Y = pca_Y.fit_transform(Y)

        # Optional Gaussian noise
        if self.sigma > 0:
            Y += self.sigma * np.random.randn(*Y.shape)

        if self.n is None:
            return X, Y
        elif self.n >= n_train:
            raise ValueError(f"Requested {self.n} training samples, but dataset only has {n_train}.")
        return train_test_split(X, Y, train_size= self.n / n_train, random_state=self.random_state)

    def _load_dataset(self):
        if self.dataset_name == "energy":
            return self._load_energy()
        elif self.dataset_name == "wine":
            return self._load_wine()
        elif self.dataset_name == "california":
            return self._load_california()
        elif self.dataset_name == "diabetes":
            return self._load_diabetes()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _load_energy(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        df = pd.read_excel(url)
        X = df.iloc[:, :8].values
        Y = df.iloc[:, 8:10].values  # Heating and Cooling Load
        return X, Y

    def _load_wine(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        df = pd.read_csv(url, sep=";")
        X = df.drop("quality", axis=1).values  # 11 features
        Y = df[["quality"]].values             # 1 target (can be extended)
        return X, Y

    def _load_california(self):
        data = fetch_california_housing()
        X = data.data
        Y = data.target.reshape(-1, 1)
        return X, Y

    def _load_diabetes(self):
        data = load_diabetes()
        X = data.data
        Y = data.target.reshape(-1, 1)
        return X, Y