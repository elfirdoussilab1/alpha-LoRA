# In this file, we will show that theoretical risk match well the empirical one
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rmt_results import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})
# Parameters
N = 1000
n = 500
p = 200
mu = 1.5
mu_orth = 1
alpha = 0
beta = 0.8
gamma_pre = 1
classifier = 'ft'

batch = 10
gammas_ft = np.logspace(-6, 3, 20)
means_practice = []
means_theory = []
for gamma_ft in tqdm(gammas_ft):
    # Theory
    means_theory.append(test_risk(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft))

    # Empirical
    means_practice.append(empirical_risk(classifier, batch, N, n, p, mu, mu_orth, beta, alpha, gamma_pre, gamma_ft ))

# Plotting results
fig, ax = plt.subplots()
ax.semilogx(gammas_ft, means_theory, label = 'Theory', color = 'purple', linewidth = 3)
ax.scatter(gammas_ft, means_practice, label = 'Simulation', marker = 'D', alpha = .7, color = 'green')
ax.set_xlabel('$\gamma$')
ax.set_ylabel('Test Risk')
ax.grid(True)
ax.legend()
path = './results-plot/' + f'simulate_risk-N-{N}-n-{n}-p-{p}-alpha-{alpha}-beta-{beta}-mu-{mu}-mu_orth-{mu_orth}-gamma_pre-{gamma_pre}.pdf'
fig.savefig(path, bbox_inches='tight')