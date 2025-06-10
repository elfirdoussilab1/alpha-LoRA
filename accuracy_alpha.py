# In this file, we verify our alpha_max obtained theoretically
import numpy as np
import matplotlib.pyplot as plt
#from utils import *
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})
# Parameters
N = 5000
n = 40
p = 400
mu = 0.7
mu_orth = 0.7
gamma_pre = 1
gamma_ft = 1

betas = [0.2, 0.5, 0.9]
linewidth = 4
fontsize = 40
labelsize = 35
s = 250
alphas = np.linspace(-10, 10, 1000)
fig, ax = plt.subplots(1, 3, figsize = (30, 6))

for i, beta in enumerate(tqdm(betas)):
    accs = []
    for alpha in alphas:
        accs.append(test_accuracy(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft))
    
    # xmax, ymax
    alpha_max, alpha_min = optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft)

    acc_max = test_accuracy(N, n, p, mu, mu_orth, alpha_max, beta, gamma_pre, gamma_ft)
    acc_min = test_accuracy(N, n, p, mu, mu_orth, alpha_min, beta, gamma_pre, gamma_ft)
    acc_zero = test_accuracy(N, n, p, mu, mu_orth, 0, beta, gamma_pre, gamma_ft)
    # LoRA
    acc_lora = test_accuracy(N, n, p, mu, mu_orth, 1, beta, gamma_pre, gamma_ft)

    ax[i].plot(alphas, accs, linewidth = linewidth, color = 'tab:blue')
    ax[i].plot([alphas[0], alphas[-1]], [acc_zero, acc_zero], linewidth = linewidth, color = 'tab:orange', linestyle = '-.')
    ax[i].plot([alphas[0], alphas[-1]], [acc_lora, acc_lora], linewidth = linewidth, color = 'tab:purple', linestyle = '-.')
    ax[i].scatter(alpha_max, acc_max, color = 'tab:green', s = s, marker = 'D')
    ax[i].scatter(alpha_min, acc_min, color = 'tab:red', s = s, marker = 'D')
    sentence_max = f'$\\alpha^*= {round(alpha_max, 2)}$'
    sentence_min = f'$ \\bar \\alpha= {round(alpha_min, 2)}$'
    #hx = 2e-3
    hx = 2
    hy = -2e-3
    ax[i].text(alpha_max+hx, acc_max+hy, sentence_max, fontsize = fontsize - 10)
    ax[i].text(alpha_min+hx, acc_min+hy, sentence_min, fontsize = fontsize - 10)
    ax[i].set_title(f'$ \\beta = {beta}$', fontsize = fontsize)
    
    ax[i].set_xlabel('$\\alpha$', fontsize = fontsize)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].grid()

ax[0].set_ylabel('Test Accuracy', fontsize = fontsize)
path = './study-plot/' + f'optimize-N-{N}-n-{n}-p-{p}-mu-{mu}-mu_orth-{mu_orth}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}.pdf'
fig.savefig(path, bbox_inches='tight')
