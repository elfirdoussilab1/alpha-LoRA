# This file contains functions to load and preprocess our datasets
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
import torch, os, random
import pandas as pd
from typing import List, Union
from tqdm.auto import tqdm

type_to_path = {
    'book' : './datasets/Amazon_review/books.mat',
    'dvd' : './datasets/Amazon_review/dvd.mat',
    'elec' : './datasets/Amazon_review/elec.mat',
    'kitchen' : './datasets/Amazon_review/kitchen.mat',
    'sentiment_train': './datasets/GPT2/sentiment_train.mat',
    'sentiment_test': './datasets/GPT2/sentiment_test.mat',
    'sentiment': './datasets/GPT2/sentiment.mat',
    'safety': './datasets/GPT2/safety.mat'

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

############ Sentiment Analysis/Safety datasets #######

def get_embedding_bert(prompt): 
    model_name = 'bert-base-uncased'
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    
    # Extract the embeddings
    prompt_embedding = embeddings[0].mean(dim=0)
    
    return prompt_embedding
    
    
# Function to get the embedding for a chunk of text
def get_chunk_embedding(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]
    chunk_embedding = last_hidden_states.mean(dim=1)
    return chunk_embedding.cpu().numpy()
    
def get_embedding_gpt2(prompt, device, max_length = 1024, overlap = 200):
    # Load pre-trained model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name).to(device)

    # Encode the input prompt
    tokens = tokenizer.encode(prompt)

    if len(tokens) <= max_length:
        return get_chunk_embedding(prompt, model, tokenizer, device)

    embeddings = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunk_text = tokenizer.decode(chunk)
        embedding = get_chunk_embedding(chunk_text, model, tokenizer, device)
        embeddings.append(embedding)
        if i + max_length >= len(tokens):  # if the last chunk, break the loop
            break

    # Combine embeddings
    combined_embedding = np.mean(embeddings, axis=0)  # You can use other aggregation methods
    return combined_embedding

class GPT2Embedder:
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(device)
        self.device = device
        self.max_length = self.model.config.n_positions

    def get_embeddings(self, texts: Union[str, List[str]], batch_size: int = 8) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self._process_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def _process_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        batch_embeddings = []

        for text in batch_texts:
            chunks = self._split_into_chunks(text)
            chunk_embeddings = []

            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Use the mean of the last hidden state as the embedding
                embedding = outputs.last_hidden_state[-1].cpu().numpy()
                chunk_embeddings.append(embedding[0])

            # Combine chunk embeddings (mean pooling)
            text_embedding = np.mean(chunk_embeddings, axis=0)
            batch_embeddings.append(text_embedding)

        return batch_embeddings

    def _split_into_chunks(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        chunks = []

        for i in range(0, len(tokens), self.max_length):
            chunk = tokens[i:i+self.max_length]
            chunks.append(self.tokenizer.convert_tokens_to_string(chunk))

        return chunks


class LLM_dataset:
    def __init__(self, n, type_name, classifier = 'pre') -> None:
        # type_name (str): either 'sentiment' or 'safety'
        data = loadmat(type_to_path[type_name])
        self.X = data['embeddings'] # shape (n, p)
        self.y = data['labels'][0].astype(int)

        # Change the order randomly (for safety)

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