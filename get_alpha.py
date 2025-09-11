import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'roberta-base'
task_name = 'rte'
rank = 8

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    ).to(device)
# Apply LoRA
apply_adapter(model, model_name, lora = True, rank = rank, alpha= 1, alpha_r= rank, device =device, train_alpha = True)
model.load_state_dict(torch.load(f'models/{model_name}_{task_name}_alpha_trainable_True.pth'))

alpha_params = []
for name, param in model.named_parameters():
    if 'alpha' in name:
        alpha_params = alpha_params + list(param.view(-1).detach().cpu().numpy())
alpha_params = np.array(alpha_params)

# Save vector
np.save('alpha_vec.npy', alpha_params)

    