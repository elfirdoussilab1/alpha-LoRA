import torch
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'roberta-base'
task_name = 'mnli'

path = f'models/{model_name}_{task_name}_alpha_trainable_True.pth'
# Load checkpoint
checkpoint = torch.load(path, map_location= device)

# Depending on how it was saved:
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

# Collect all 'alpha' params
alpha_params = []
for name, param in state_dict.items():
    if 'alpha' in name:
        alpha_params = alpha_params + list(param.view(-1).detach().cpu().numpy())
alpha_params = np.array(alpha_params)

# Save vector
np.save(f'alpha_vec_{task_name}.npy', alpha_params)
