# In this file, we will create the model that we will use to generate our embeddings
# Our model will be trained on the sentiment analysis task
import torch
import torch.nn as nn
import math
import numpy as np

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

class Adapter(nn.Module):
    def __init__(self, linear, lora= False, rank = 8, alpha = 1, alpha_r = 8, train_alpha = False):
        super().__init__()
        # These are the weights from the original pretrained model
        self.linear = linear
        in_dim = linear.in_features
        out_dim = linear.out_features
        self.alpha_r = alpha_r # This is the old alpha used in Pytorch
        self.rank = rank 
        self.lora = lora

        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Parameter alpha
        if alpha is None:
            alpha_v = np.random.randn(out_dim) # shape (out_dim, )
        else:
            alpha_v = alpha * np.ones(out_dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_v, dtype = torch.float), requires_grad = train_alpha) # This is our alpha parameter in the theory

        if lora:
            # These are the LoRA parameters
            std = 1 / math.sqrt(rank)
            self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * std, requires_grad= True)
            self.lora_B = nn.Parameter(torch.zeros(rank, out_dim), requires_grad= True)
        else: # Classical Adapter
            self.adapter = nn.Linear(in_dim, out_dim) # requires_grad is set to True here by default
            # Initialize to zero
            nn.init.zeros_(self.adapter.weight)
            nn.init.zeros_(self.adapter.bias)
    
    def forward(self, x):
        scaled_output = self.alpha * self.linear(x) 
        # Adapter update
        if self.lora:
            update = (x @ self.lora_A @ self.lora_B) * self.alpha_r / self.rank
        else: 
            update = self.adapter(x)
        return scaled_output + update

def replace_linear_with_adapter(model, lora, rank, alpha, alpha_r, device, train_alpha=False):
    """
    Replaces all nn.Linear layers in a model with LoRALinear layers,
    explicitly skipping nn.Embedding layers.
    """
    # Freeze all original model weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Recursive function to replace layers
    def _replace(module):
        for name, child in module.named_children():
            # If the child is a Linear layer, replace it with a LoRALinear layer
            if isinstance(child, nn.Linear):
                adapter_layer = Adapter(
                    child,
                    lora, 
                    rank,
                    alpha=alpha,
                    alpha_r=alpha_r,
                    train_alpha=train_alpha
                ).to(device)
                setattr(module, name, adapter_layer)

            # Explicitly skip embedding layers and do not recurse into them
            elif isinstance(child, nn.Embedding):
                continue
            
            # For all other module types, recurse
            else:
                _replace(child)

    _replace(model)

def optimize_adapter(model):
    # The model given as input should only have lora weights trainable
    # Additionally, this method is designed for the case of alpha trainable
    adapter_params = []
    alpha_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'alpha' in name:
                alpha_params.append(param)
            else: # Adapter weights
                adapter_params.append(param)
    return adapter_params, alpha_params

def make_adapter_roberta(model, lora, rank, alpha, alpha_r, device, train_alpha=False):
    # For Roberta-base model, we add LoRA weights only for the query and value weights
    # Freeze all original model weights except the last classififer layer
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True
    # Recursive function to replace layers
    def _replace(module):
        for name, child in module.named_children():
            # If the child is a Linear layer, replace it with a LoRALinear layer
            if 'query' in name or 'value' in name:
                if isinstance(child, nn.Linear):
                    adapter_layer = Adapter(
                        child,
                        lora, 
                        rank,
                        alpha=alpha,
                        alpha_r=alpha_r,
                        train_alpha=train_alpha
                    ).to(device)
                    setattr(module, name, adapter_layer)
            
            # For all other module types, recurse
            else:
                _replace(child)

    _replace(model)

def apply_adapter(model, model_name, lora, rank, alpha, alpha_r, device, train_alpha=False):
    if "roberta" in model_name:
        make_adapter_roberta(model, lora, rank, alpha, alpha_r, device, train_alpha)
    
    else: # Apply LoRA to all Linear layers
        replace_linear_with_adapter(model, lora, rank, alpha, alpha_r, device, train_alpha)


# Changing only the alphas in the Adapter modules
def change_alpha(adapter_model, new_alpha):
    for name, param in adapter_model.named_parameters():
        if 'alpha' in name:
            param.data = new_alpha * torch.ones_like(param.data, dtype=torch.float)
            param.requires_grad = False

# Get the value of certain alpha
def get_alpha(model, model_name):
    if "roberta" in model_name:
        return model.roberta.encoder.layer[0].attention.self.query.alpha[0].detach().cpu().numpy()
    else:
        return model.classifier.alpha[0].detach().cpu().numpy()
    
