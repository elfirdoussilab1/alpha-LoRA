# In this file, we verify our alpha_max obtained theoretically
import numpy as np
import matplotlib.pyplot as plt
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

alphas = np.linspace(-4, 4, 81)
batch = 10

fig, ax = plt.subplots(1, 3, figsize = (30, 5))
linewidth = 4.5
fontsize = 40
labelsize = 35
s = 200

seeds = [123, 404, 1337]
for i, (source, target) in enumerate(data_source_target):
    data_type = 'amazon_' + source + '_' + target

    # Datasets
    data_pre, data_ft, beta, vmu_orth = dataset.create_pre_ft_datasets(N, source, n, target, dataset_name)
    mu_orth = np.linalg.norm(vmu_orth)
    mu = data_pre.mu
    mu_orth = np.linalg.norm(vmu_orth)
    accs = []
    for seed in seeds:
        fix_seed(seed)
        accs_i = []
        for alpha in tqdm(alphas):
            accs_i.append(empirical_accuracy('ft', batch, N, n, p, mu, mu_orth, beta, alpha, gamma_pre, gamma_ft, data_type= data_type))
        accs.append(accs_i)   
    accs = np.mean(accs, axis = 0)

    # xmax, ymax
    #alpha_max, alpha_min = optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft)
    alpha_max = alphas[np.argmax(accs)]
    acc_max = np.max(accs)
    acc_zero = accs[np.where(alphas == 0)[0][0]]
    acc_one = accs[np.where(alphas == 1)[0][0]]

    ax[i].plot(alphas, accs, linewidth = linewidth, color = 'tab:blue')
    ax[i].plot([alphas[0], alphas[-1]], [acc_zero, acc_zero], linewidth = linewidth, color = 'tab:orange', linestyle = '-.')
    ax[i].scatter(1, acc_one, color = 'tab:purple', marker= 'D', s = s)
    #ax[i].plot([alphas[0], alphas[-1]], [acc_one, acc_one], linewidth = 2.5, color = 'tab:purple', linestyle = '-.')
    ax[i].scatter(alpha_max, acc_max, color = 'tab:green', marker = 'D', s = s)
    sentence_max = f'$\\alpha^*= {round(alpha_max, 2)}$'
    #sentence_min = f'$ \\bar \\alpha= {round(alpha_min, 2)}$'
    #hx = 2e-3
    hx = 1e-2
    hy = 2e-3
    #ax[i].text(alpha_max+hx, acc_max+hy, sentence_max)
    #ax[i].text(alpha_min+hx, acc_min+hy, sentence_min)
    ax[i].set_title(f'{source.capitalize()} To {target.capitalize()}', fontsize = fontsize)
    ax[i].set_xlabel('$\\alpha$', fontsize = fontsize)
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].grid()

ax[0].set_ylabel('Test Accuracy', fontsize = fontsize)
path = './study-plot/' + f'accuracy_alpha_real-amazon-N-{N}-n-{n}-p-{p}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}.pdf'
fig.savefig(path, bbox_inches='tight')
