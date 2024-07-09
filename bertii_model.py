# In this file, we will create the model that we will use to generate our embeddings
# Our model will be trained on the sentiment analysis task
import torch
import torch.nn as nn

class BerTII(nn.Module):
    def __init__(self, p, vocab_size):
        # p is the Embedding dimension, we get to choose it
        super().__init__()
        self.p = p
        self.vocab_size = vocab_size
        self.embedding_table = nn.Embedding(vocab_size, p)
        self.ln = nn.LayerNorm(p)
        self.linear = nn.Linear(p, 1)
    
    def forward(self, X, N):
        # X.shape : (B, context)
        X = self.embedding_table(X) # (B, context, p)
        #X = torch.tensor(torch.nested.to_padded_tensor(X, 0.0), dtype = torch.float)
        X = torch.nested.to_padded_tensor(X, 0.0).clone().detach()
        X = torch.sum(X, dim = 1) # (B, p)
        X = X / N.unsqueeze(1) # divide each row (prompt) by its length
        X = self.linear(self.ln(X)) # (B, 1)
        logits = torch.sigmoid(X) # (B, 1)
        B = logits.shape[0]
        return logits.view(B) # (B, )

class LinearWithFTLayer(nn.Module):
    def __init__(self, linear, p, alpha = 1):
        super().__init__()
        #self.alpha = torch.nn.Parameter(torch.tensor(1, dtype = torch.float, requires_grad= True))
        self.alpha = alpha
        #self.alpha.requires_grad = True
        self.linear = linear
        self.V = nn.Linear(p, 1, bias = False)

    def forward(self, x):
        x = self.alpha * self.linear(x) + self.V(x)
        return x

def replace_linear_with_ft(model, p):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace the linear layer with LinearWithFTLayer
            setattr(model, name, LinearWithFTLayer(module, p))
        else:
            continue
