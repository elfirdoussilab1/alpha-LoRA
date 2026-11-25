# In this file, we verify our alpha_max obtained theoretically
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from utils import *
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})

# Parameters
N = 2000
n = 40
p = 400
gamma_pre = 2
gamma_ft = 1e-1
dataset_name = 'amazon'
data_source_target = [('dvd', 'book'), ('book', 'kitchen'), ('kitchen', 'elec')]

pad = 0.15         # margin on each side of [min(alpha_bar, 0), max(alpha^*, 1)], as a fraction of that span (as in accuracy_alpha.py)
n_alphas = 81      # number of alpha values (each one costs len(seeds) * batch fits)
batch = 10

fig, ax = plt.subplots(1, 3, figsize = (30, 6))
linewidth = 4
fontsize = 40
labelsize = 35
s = 250

seeds = [123, 404, 1337]
for i, (source, target) in enumerate(data_source_target):
    data_type = 'amazon_' + source + '_' + target

    # Datasets
    data_pre, data_ft, beta, vmu_orth = dataset.create_pre_ft_datasets(N, source, n, target, dataset_name)
    mu_orth = np.linalg.norm(vmu_orth)
    mu = data_pre.mu

    # adaptive x-range from the theoretical alpha^* and alpha_bar (same rule as accuracy_alpha.py), computed once
    a_th, abar_th = optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft)
    lo = min(abar_th, a_th, 0, 1)
    hi = max(abar_th, a_th, 0, 1)
    span = hi - lo
    alphas = np.linspace(lo - pad * span, max(hi + pad * span, min(2.5 * a_th, a_th + 8)), n_alphas)
    alphas = np.union1d(alphas, [0, 1])      # make sure alpha = 0 and alpha = 1 are on the grid

    accs = []
    for seed in seeds:
        fix_seed(seed)
        accs_i = []
        for alpha in tqdm(alphas):
            accs_i.append(empirical_accuracy('ft', batch, N, n, p, mu, mu_orth, beta, alpha, gamma_pre, gamma_ft, data_type= data_type))
        accs.append(accs_i)
    accs = np.mean(accs, axis = 0)

    # xmax, ymax
    alpha_max = alphas[np.argmax(accs)]
    acc_max = np.max(accs)
    acc_zero = accs[np.argmin(np.abs(alphas - 0))]
    acc_one = accs[np.argmin(np.abs(alphas - 1))]

    ax[i].plot(alphas, accs, linewidth = linewidth, color = 'tab:blue')
    ax[i].plot([alphas[0], alphas[-1]], [acc_zero, acc_zero], linewidth = linewidth, color = 'tab:orange', linestyle = '-.')
    ax[i].scatter(1, acc_one, color = 'tab:purple', marker= 'D', s = s)
    ax[i].scatter(alpha_max, acc_max, color = 'tab:green', marker = 'D', s = s)
    sentence_max = f'$\\alpha^*= {round(alpha_max, 2)}$'
    dy = 0.05 * (max(accs) - min(accs))      # alpha^* label centred above its marker (headroom added below)
    ax[i].text(alpha_max, acc_max + dy, sentence_max, fontsize = fontsize - 10, ha = 'center', va = 'bottom')
    ax[i].set_title(f'{source.capitalize()} To {target.capitalize()}', fontsize = fontsize)

    ax[i].set_xlim(alphas[0], alphas[-1])
    ax[i].set_xlabel('$\\alpha$', fontsize = fontsize)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    # finer grid: more major ticks + 4 minor subdivisions, minor lines lighter
    ax[i].xaxis.set_major_locator(MaxNLocator(nbins = 12, steps = [1, 2, 5, 10]))
    ax[i].yaxis.set_major_locator(MaxNLocator(nbins = 8, steps = [1, 2, 2.5, 5, 10]))
    ax[i].set_ylim(min(accs) - dy, acc_max + 5 * dy)
    yt = ax[i].get_yticks()                   # no tick label above 100% accuracy
    ax[i].set_yticks(yt[yt <= 1.0 + 1e-9])
    ax[i].set_ylim(min(accs) - dy, acc_max + 5 * dy)
    ax[i].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[i].yaxis.set_minor_locator(AutoMinorLocator(4))
    ax[i].grid(which = 'major', linewidth = 1.2, alpha = 0.6)
    ax[i].grid(which = 'minor', linewidth = 0.6, alpha = 0.3, linestyle = ':')

ax[0].set_ylabel('Test Accuracy', fontsize = fontsize)
path = './study-plot/' + f'accuracy_alpha_real-amazon-N-{N}-n-{n}-p-{p}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}.pdf'
fig.savefig(path, bbox_inches='tight')
