# In this file, we will show that theoretical risk match well the empirical one
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from rmt_results import *
from utils import *
from matplotlib.ticker import ScalarFormatter

fix_seed(123)
plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})
# Parameters
n = 20
p = 200
d = 4
sigma = 0.5

W_s = np.random.randn(d, p) 
gamma = 1e-2

batch = 10
fig, ax = plt.subplots(1, 3, figsize = (30, 6))
linewidth = 4.5
fontsize = 35
labelsize = 30
s = 200

betas = [-2, 0.5, 2]
alphas = np.linspace(-4, 4, 20)

for i, beta in enumerate(betas):
    B = generate_frobenius_orthogonal_matrix(W_s)
    W_t = beta * W_s + B 
    error_practice = []
    error_theory = []
    alpha_opt = optimal_alpha_regression(W_s, W_t)

    print(f"Optimal alpha is: {alpha_opt}")
    for alpha in tqdm(alphas):
        # Theory
        error_theory.append(test_risk_regression(n, p, d, sigma, alpha, W_s, W_t, gamma))

        # Empirical
        error_practice.append(empirical_risk_regression(batch, n, p, d, sigma, alpha, W_s, W_t, gamma))

    min_err = test_risk_regression(n, p, d, sigma, alpha_opt, W_s, W_t, gamma)
    # Plotting results
    ax[i].plot(alphas, error_theory, label = 'Theory', color = 'tab:blue', linewidth = linewidth)
    ax[i].scatter(alphas, error_practice, label = 'Simulation', marker = 'o', alpha = .7, color = 'tab:red', s = s)
    ax[i].scatter(alpha_opt, np.min(error_theory), marker = 'D', color = 'tab:green', s = s, label = 'Optimal $\\alpha^*$')
    ax[i].set_title(f'$\\beta = {beta}$', fontsize = fontsize)
    ax[i].set_xlabel('$\\alpha$', fontsize = fontsize)
    sentence_max = f'$\\alpha^*= {round(alpha_opt, 2)}$'
    hx = -1
    hy = 3000
    ax[i].text(alpha_opt+hx, min_err+hy, sentence_max, fontsize = labelsize)
    # Format y-axis in scientific notation
    ax[i].tick_params(axis='x', which = 'both', labelsize=labelsize)
    ax[i].tick_params(axis='y', which = 'both', labelsize=labelsize)
    ax[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[i].yaxis.offsetText.set_fontsize(labelsize)
    ax[i].grid()
ax[0].legend(fontsize = labelsize)
ax[0].set_ylabel('Test Risk', fontsize = fontsize)
path = './study-plot/' + f'simulate_risk_regresion-n-{n}-p-{p}-d-{d}-gamma-{gamma}-sigma-{sigma}.pdf'
fig.savefig(path, bbox_inches='tight')