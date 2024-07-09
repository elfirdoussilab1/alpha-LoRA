# In this file, we will evaluate our Fine-Tuning method using either L2 or BCELoss.
import numpy as np
import matplotlib.pyplot as plt
from dataset import *
from tqdm.auto import tqdm
from utils import *

#plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})

# Parameters
n = 500
p = 1000
N = 2000
gamma_pre = 1e-1
gamma_ft = 1e-1
batch = 1

# Using L^2 solutions
alphas = np.linspace(-10, 10, 20)
seeds = [1, 123, 404]
dataset_name = 'llm'
source = 'sentiment'
target = 'safety'
data_type = 'llm_sentiment_safety'
all_accs = []
accs_no_ft = []
for seed in seeds:
    accs = []
    for alpha in tqdm(alphas):
        accs.append(empirical_accuracy('ft', batch, N, n, p, None, None, None, alpha, gamma_pre, gamma_ft, data_type))

    accs_no_ft.append(empirical_accuracy('ft', batch, N, n, p, None, None, None, 0, gamma_pre, gamma_ft, data_type))
    all_accs.append(accs)

np.save('all_accs.npy', np.array(all_accs))
# Plotting results
linewidth = 3

plt.plot(alphas, np.mean(all_accs, axis = 0), color = 'tab:blue', linewidth = linewidth, label = '$\\alpha$-FT')
plt.fill_between(alphas, np.mean(all_accs, axis = 0) - np.std(all_accs, axis = 0), np.mean(all_accs, axis = 0) + np.std(all_accs, axis = 0), 
         alpha = 0.2, linestyle = '-.', color = 'tab:blue')

# No-FT accuracy
plt.plot([alphas[0], alphas[-1]], [np.mean(accs_no_ft), np.mean(accs_no_ft)], '-.', color = 'tab:purple', linewidth = linewidth, label = 'No-FT')
plt.fill_between([alphas[0], alphas[-1]], [np.mean(accs_no_ft) - np.std(accs_no_ft), np.mean(accs_no_ft) - np.std(accs_no_ft)], 
         [np.mean(accs_no_ft) + np.std(accs_no_ft), np.mean(accs_no_ft) + np.std(accs_no_ft)], 
         alpha = 0.2, linestyle = '-.', color = 'tab:orange')

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










