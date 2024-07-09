# In this file, we will evaluate our Fine-Tuning method using either L2 or BCELoss.
import numpy as np
import matplotlib.pyplot as plt
from dataset import *
from tqdm.auto import tqdm
from utils import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})

# Parameters
n = 20
p = 1000
N = 5000
gamma_pre = 1e-2
gamma_ft = 1e-2
batch = 1

# Using L^2 solutions
alphas = np.linspace(-20, 20, 50)
seeds = [1, 123, 404]
source = 'sentiment'
target = 'safety'
data_type = 'llm_sentiment_safety'
all_accs = []
accs_no_ft = []
accs_pre = []
for seed in seeds:
    fix_seed(seed)
    accs = []
    for alpha in tqdm(alphas):
        accs.append(empirical_accuracy('ft', batch, N, n, p, None, None, None, alpha, gamma_pre, gamma_ft, data_type))

    all_accs.append(accs)

    fix_seed(seed)
    accs_no_ft.append(empirical_accuracy('ft', batch, N, n, p, None, None, None, 0, gamma_pre, gamma_ft, data_type))
    accs_pre.append(empirical_accuracy('pre', batch, N, n, p,  None, None, None, None, gamma_pre= gamma_pre, gamma_ft= gamma_ft, data_type=data_type))
    
np.save('all_accs.npy', np.array(all_accs))
# Plotting results
linewidth = 2.

plt.plot(alphas, np.mean(all_accs, axis = 0), color = 'tab:blue', linewidth = linewidth, label = '$\\alpha$-FT')
plt.fill_between(alphas, np.mean(all_accs, axis = 0) - np.std(all_accs, axis = 0), np.mean(all_accs, axis = 0) + np.std(all_accs, axis = 0), 
         alpha = 0.2, linestyle = '-.', color = 'tab:blue')

# No-FT accuracy
plt.plot([alphas[0], alphas[-1]], [np.mean(accs_no_ft), np.mean(accs_no_ft)], '-.', color = 'tab:orange', linewidth = linewidth, label = 'No-FT')
plt.fill_between([alphas[0], alphas[-1]], [np.mean(accs_no_ft) - np.std(accs_no_ft), np.mean(accs_no_ft) - np.std(accs_no_ft)], 
         [np.mean(accs_no_ft) + np.std(accs_no_ft), np.mean(accs_no_ft) + np.std(accs_no_ft)], 
         alpha = 0.2, linestyle = '-.', color = 'tab:orange')

# Pre accuracy
plt.plot([alphas[0], alphas[-1]], [np.mean(accs_pre), np.mean(accs_pre)], '-.', color = 'tab:brown', linewidth = linewidth, label = 'Pre-trained')
plt.fill_between([alphas[0], alphas[-1]], [np.mean(accs_pre) - np.std(accs_pre), np.mean(accs_pre) - np.std(accs_pre)], 
         [np.mean(accs_pre) + np.std(accs_pre), np.mean(accs_pre) + np.std(accs_pre)], 
         alpha = 0.2, linestyle = '-.', color = 'tab:brown')

# Max and Min points
x_max, y_max = alphas[np.argmax(np.mean(all_accs, axis = 0))], np.max(np.mean(all_accs, axis = 0))
x_min, y_min = alphas[np.argmin(np.mean(all_accs, axis = 0))], np.min(np.mean(all_accs, axis = 0))
plt.scatter(x_max, y_max, color = 'tab:green', marker = 'D')
plt.scatter(x_min, y_min, color = 'tab:red', marker = 'D')
sentence_max = f'$\\alpha^*= {round(x_max, 2)}$'
sentence_min = f'$\\bar \\alpha= {round(x_min, 2)}$'
plt.text(x_max - 5e-2, y_max-  5e-2, sentence_max)
plt.text(x_min+ 3e-1, y_min , sentence_min)
plt.xlabel('$\\alpha $')
plt.ylabel('Test Accuracy')
plt.grid()
plt.legend()
path = './study-plot/' + f'evaluate_ft_L2-N-{N}-n-{n}-p-{p}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}.pdf'
plt.savefig(path, bbox_inches='tight')










