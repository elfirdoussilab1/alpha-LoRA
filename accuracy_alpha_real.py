# In this file, we verify our alpha_max obtained theoretically
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from rmt_results import *
from tqdm.auto import tqdm

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})

# Parameters MNIST
N = 2000
n = 20
p = 784
gamma_pre = 1
gamma_ft = 1
type_1 = '6_8'
type_2 = '5_9'
data_type = 'mnist_' + type_1 + '_' + type_2
dataset_name = 'mnist'
batch = 1

alphas = np.linspace(-2, 2, 100)

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
alpha_max, alpha_min = optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft)
acc_max = empirical_accuracy('ft', batch, N, n, p, mu, mu_orth, beta, alpha_max, gamma_pre, gamma_ft, data_type= data_type)
#acc_min = empirical_accuracy('ft', batch, N, n, p, mu, mu_orth, beta, alpha_min, gamma_pre, gamma_ft, data_type= data_type)
acc_zero = empirical_accuracy('ft', batch, N, n, p, mu, mu_orth, beta, 0, gamma_pre, gamma_ft, data_type= data_type)
ax.plot(alphas, accs, linewidth = 2.5, color = 'tab:blue')
ax.plot([alphas[0], alphas[-1]], [acc_zero, acc_zero], linewidth = 2.5, color = 'tab:orange', linestyle = '-.')
ax.scatter(alpha_max, acc_max, color = 'tab:green', marker = 'D')
#ax.scatter(alpha_min, acc_min, color = 'tab:red', marker = 'D')
sentence_max = f'$\\alpha^*= {round(alpha_max, 2)}$'
sentence_min = f'$ \\bar \\alpha= {round(alpha_min, 2)}$'
#hx = 2e-3
hx = 2
hy = -2e-3
ax.text(alpha_max+hx, acc_max+hy, sentence_max)
#ax.text(alpha_min+hx, acc_min+hy, sentence_min)
ax.set_title(f'{dataset_name} from {type_1} to {type_2}')

ax.set_xlabel('$\\alpha$')
ax.grid()

ax.set_ylabel('Test Accuracy')
path = './study-plot/' + f'accuracy_alpha_real-{dataset_name}-{type_1}-{type_2}-N-{N}-n-{n}-p-{p}-mu-{mu}-mu_orth-{mu_orth}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}.pdf'
fig.savefig(path, bbox_inches='tight')
