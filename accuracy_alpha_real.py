# In this file, we verify our alpha_max obtained theoretically
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})

# Parameters for MNIST
N = 2000
n = 10
p = 784
gamma_pre = 1
gamma_ft = 1
type_1 = '6_9'
type_2 = '4_9'
data_type = 'mnist_' + type_1 + '_' + type_2
dataset_name = 'mnist'
batch = 5

alphas = np.linspace(-2, 2, 41)

# Datasets
data_pre, data_ft, beta, vmu_orth = dataset.create_pre_ft_datasets(N, type_1, n, type_2, dataset_name)
mu_orth = np.linalg.norm(vmu_orth)
mu = data_pre.mu
mu_orth = np.linalg.norm(vmu_orth)

accs = []
for alpha in tqdm(alphas):
    accs.append(empirical_accuracy('ft', batch, N, n, p, mu, mu_orth, beta, alpha, gamma_pre, gamma_ft, data_type= data_type))

fig, ax = plt.subplots()
# xmax, ymax
#alpha_max, alpha_min = optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft)
alpha_max = alphas[np.argmax(accs)]
acc_max = np.max(accs)
acc_zero = accs[np.where(alphas == 0)[0][0]]
acc_one = accs[np.where(alphas == 1)[0][0]]

ax.plot(alphas, accs, linewidth = 2.5, color = 'tab:blue')
ax.plot([alphas[0], alphas[-1]], [acc_zero, acc_zero], linewidth = 2.5, color = 'tab:orange', linestyle = '-.')
ax.plot([alphas[0], alphas[-1]], [acc_one, acc_one], linewidth = 2.5, color = 'tab:purple', linestyle = '-.')
ax.scatter(alpha_max, acc_max, color = 'tab:green', marker = 'D')
sentence_max = f'$\\alpha^*= {round(alpha_max, 2)}$'
#sentence_min = f'$ \\bar \\alpha= {round(alpha_min, 2)}$'
#hx = 2e-3
hx = 1e-2
hy = 2e-3
ax.text(alpha_max+hx, acc_max+hy, sentence_max)
#ax.text(alpha_min+hx, acc_min+hy, sentence_min)
ax.set_title(f'{dataset_name} from {type_1} to {type_2}, $\\beta = {beta}$')
ax.set_xlabel('$\\alpha$')
ax.grid()

ax.set_ylabel('Test Accuracy')
path = './study-plot/' + f'accuracy_alpha_real-{dataset_name}-{type_1}-{type_2}-N-{N}-n-{n}-p-{p}-mu-{mu}-mu_orth-{mu_orth}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}.pdf'
fig.savefig(path, bbox_inches='tight')
