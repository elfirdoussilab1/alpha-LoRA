# In this file, we will create the model that we will use to generate our embeddings
# Our model will be trained on the sentiment analysis task
import torch
import torch.nn as nn
import math

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

# Simple MNIST model
# Binary classification
class simple_mnist(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.linear_1 = nn.Linear(784, p)
        self.linear_2 = nn.Linear(p, 1)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = torch.tanh(x)
        x = self.linear_2(x)
        logits = torch.sigmoid(x) # (B, 1)
        B = logits.shape[0]
        return logits.view(B, )

class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, alpha_r):
        super().__init__()
        # These are the weights from the original pretrained model
        self.linear = linear
        in_dim = linear.in_features
        out_dim = linear.out_features
        
        # These are the LoRA parameters
        std = 1 / math.sqrt(rank)
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * std, requires_grad= True)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim), requires_grad= True)
        
        # Other parameters of lora
        self.rank = rank 
        self.alpha = alpha # This is our alpha parameter in the theory
        self.alpha_r = alpha_r # This is the old alpha used in Pytorch
        # we can also set: self.alpha = nn.Parameter(..) if we want to make it trainable
    
    def forward(self, x):
        x = self.linear(self.alpha * x) +  (x @ self.lora_A @ self.lora_B) * self.alpha_r / self.rank
        return x

def replace_linear_with_lora(model, rank, alpha, alpha_r, device):
    # Freeze the model weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Now replace linear with LoRA weights
    def _replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Create a new LoRALinear instance
                lora_layer = LoRALinear(child,
                    rank,
                    alpha=alpha,
                    alpha_r=alpha_r  # only if your implementation uses it
                ).to(device)

                # Replace the linear layer with the LoRA version
                setattr(module, name, lora_layer)

            else:
                # Recursively apply to child modules
                _replace(child)

    _replace(model)
           