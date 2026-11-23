# In this file, we verify our alpha_max obtained theoretically
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
#from utils import *
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})
# Parameters
N = 5000
n = 40
p = 1000
mu = 1.5
mu_orth = 1.5
gamma_pre = 20
gamma_ft = 1e-1

betas = [0.2, 0.5, 0.9]
linewidth = 4
fontsize = 40
labelsize = 35
s = 250
pad = 0.15         # margin on each side of [min(alpha_bar, 0), max(alpha^*, 1)], as a fraction of that span
n_alphas = 1000
fig, ax = plt.subplots(1, 3, figsize = (30, 6))

for i, beta in enumerate(tqdm(betas)):
    # xmax, ymax
    alpha_max, alpha_min = optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft)

    # adaptive x-range: always shows alpha_bar, 0 (No-FT), 1 (LoRA) and alpha^*
    lo = min(alpha_min, alpha_max, 0, 1)
    hi = max(alpha_min, alpha_max, 0, 1)
    span = hi - lo
    alphas = np.linspace(lo - pad * span, max(hi + pad * span, min(2.5 * alpha_max, alpha_max + 8)), n_alphas)   # extend past alpha^* (at most +8) to show the decline

    accs = []
    for alpha in alphas:
        accs.append(test_accuracy(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft))

    acc_max = test_accuracy(N, n, p, mu, mu_orth, alpha_max, beta, gamma_pre, gamma_ft)
    acc_min = test_accuracy(N, n, p, mu, mu_orth, alpha_min, beta, gamma_pre, gamma_ft)
    acc_zero = test_accuracy(N, n, p, mu, mu_orth, 0, beta, gamma_pre, gamma_ft)
    # LoRA
    acc_lora = test_accuracy(N, n, p, mu, mu_orth, 1, beta, gamma_pre, gamma_ft)

    ax[i].plot(alphas, accs, linewidth = linewidth, color = 'tab:blue')
    ax[i].plot([alphas[0], alphas[-1]], [acc_zero, acc_zero], linewidth = linewidth, color = 'tab:orange', linestyle = '-.')
    #ax[i].plot([alphas[0], alphas[-1]], [acc_lora, acc_lora], linewidth = linewidth, color = 'tab:purple', linestyle = '-.')
    ax[i].scatter(alpha_max, acc_max, color = 'tab:green', s = s, marker = 'D')
    ax[i].scatter(alpha_min, acc_min, color = 'tab:red', s = s, marker = 'D')
    ax[i].scatter(1, acc_lora, color = 'tab:purple', s = s, marker = 'D')
    sentence_max = f'$\\alpha^*= {round(alpha_max, 2)}$'
    sentence_min = f'$ \\bar \\alpha= {round(alpha_min, 2)}$'
    hx = 0.04 * (alphas[-1] - alphas[0])     # label offset scales with the x-range
    hy = -2e-3
    dy = 0.05 * (max(accs) - min(accs))      # alpha^* label centred above its marker (headroom added below)
    ax[i].text(alpha_max, acc_max + dy, sentence_max, fontsize = fontsize - 10, ha = 'center', va = 'bottom')
    ax[i].set_ylim(min(accs) - dy, acc_max + 5 * dy)
    ax[i].text(alpha_min+hx, acc_min+hy, sentence_min, fontsize = fontsize - 10)
    ax[i].set_title(f'$ \\beta = {beta}$', fontsize = fontsize)
    
    ax[i].set_xlim(alphas[0], alphas[-1])
    ax[i].set_xlabel('$\\alpha$', fontsize = fontsize)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    # finer grid: more major ticks + 4 minor subdivisions, minor lines lighter
    ax[i].xaxis.set_major_locator(MaxNLocator(nbins = 12, steps = [1, 2, 5, 10]))
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins = 8, steps = [1, 2, 2.5, 5, 10]))
    yt = ax[i].get_yticks()                   # no tick label above 100% accuracy
    ax[i].set_yticks(yt[yt <= 1.0 + 1e-9])
    ax[i].set_ylim(min(accs) - dy, acc_max + 5 * dy)
    ax[i].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[i].yaxis.set_minor_locator(AutoMinorLocator(4))
    ax[i].grid(which = 'major', linewidth = 1.2, alpha = 0.6)
    ax[i].grid(which = 'minor', linewidth = 0.6, alpha = 0.3, linestyle = ':')

ax[0].set_ylabel('Test Accuracy', fontsize = fontsize)
path = './study-plot/' + f'optimize-N-{N}-n-{n}-p-{p}-mu-{mu}-mu_orth-{mu_orth}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}.pdf'
fig.savefig(path, bbox_inches='tight')
