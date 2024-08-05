# In this file, we will show that theoretical accuracy matches well the empirical one for real dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rmt_results import *
import dataset

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})
'''
# Parameters Amazon
N = 1000
n = 100
p = 400
gamma_pre = 1
gamma_ft = 1
type_1 = 'elec'
type_2 = 'kitchen'
data_type = 'amazon_' + type_1 + '_' + type_2
'''
# Parameters MNIST
N = 5000
n = 50
p = 784
gamma_pre = 1
gamma_ft = 1
type_1 = '6_8'
type_2 = '5_9'
data_type = 'mnist_' + type_1 + '_' + type_2
dataset_name = 'mnist'

# Datasets
data_pre, data_ft, beta, vmu_orth = dataset.create_pre_ft_datasets(N, type_1, n, type_2, dataset_name)
mu_orth = np.linalg.norm(vmu_orth)
mu = data_pre.mu
X_pre, y_pre = data_pre.X_train.T, data_pre.y_train
X_ft, y_ft = data_ft.X_train.T, data_ft.y_train

alpha = optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft)[0]

batch = 10
gammas_ft = np.logspace(-6, 3, 20)
accs_practice = []
accs_theory = []
for gamma_ft in tqdm(gammas_ft):
    # Theory
    accs_theory.append(test_accuracy(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft))

    # Empirical
    accs_practice.append(empirical_accuracy('ft', batch, N, n, p, mu, mu_orth, beta, alpha, gamma_pre, gamma_ft, data_type= data_type))

# Plotting results
fig, ax = plt.subplots()
ax.semilogx(gammas_ft, accs_theory, label = 'Theory', color = 'purple', linewidth = 3)
ax.scatter(gammas_ft, accs_practice, label = 'Simulation', marker = 'D', alpha = .7, color = 'green')
ax.set_xlabel('$\gamma$')
ax.set_ylabel('Test Accuracy')
ax.grid(True)
ax.legend()
path = './study-plot/' + f'simulate_accuracy_real-{dataset_name}-N-{N}-n-{n}-p-{p}-alpha-{alpha}-beta-{beta}-mu-{mu}-mu_orth-{mu_orth}-gamma_pre-{gamma_pre}.pdf'
fig.savefig(path, bbox_inches='tight')