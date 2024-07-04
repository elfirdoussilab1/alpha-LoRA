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