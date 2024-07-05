# In this file, we will implement embedders
import torch
import tiktoken
from sentiment_model import *
from typing import List, Union
from tqdm.auto import tqdm
import numpy as np

# Our embedder
class CustomEmbdedder:
    def __init__(self, p, path, device= "cuda"):
        self.p = p
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.vocab_size = self.tokenizer.max_token_value
        self.model = BerTII(p, self.vocab_size).to(device)
        self.model.load_state_dict(torch.load(path))
        self.device = device
    
    def get_embeddings(self, texts: Union[str, List[str]], batch_size: int = 8) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self._process_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)
    
    @torch.no_grad
    def _process_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        batch_embeddings = []
        
        for text in batch_texts:
            tokens = self.tokenizer.encode(text)
            x = torch.tensor(tokens, dtype= torch.long).to(self.device) # (context,)
            x = self.model.embedding_table(x) # (context, p)
            x = x.mean(dim = 0) # (p, )
            batch_embeddings.append(x.detach().cpu().numpy())
        return batch_embeddings
    
############ Old embedders #######

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
