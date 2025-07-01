# This file is used to generate distribution plots on Synthetic data to validate theoretical calculus
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

fix_seed(123)
# Parameters
p = 400
n = 200
gamma = 1

# Arbitrary classifier
w_norm = 1.
w_tilde = np.random.randn(p)
w_tilde = w_tilde / np.linalg.norm(w_tilde) * w_norm

# Arbitrary vmu_beta
mu_beta = 1.
vmu_beta = np.random.randn(p)
vmu_beta = vmu_beta / np.linalg.norm(vmu_beta) * mu_beta

# Dataset
X_train, y_train = gaussian_mixture(n, vmu_beta)
X_test, y_test = gaussian_mixture(20*n, vmu_beta)

alphas = [1e-1, 1, 10]
fig, ax = plt.subplots(1, 3, figsize = (30, 6))
fontsize = 40
labelsize = 35
linewidth = 3

for i, alpha in enumerate(alphas):
    # Expectation of class C_1 and C_2
    mean_c2 = test_expectation_arbitrary(n, p, w_tilde, vmu_beta, alpha, gamma)
    mean_c1 = - mean_c2
    expec_2 = test_expectation_2_arbitrary(n, p, w_tilde, vmu_beta, alpha, gamma)
    std = np.sqrt(expec_2 - mean_c2**2)

    # Classifier
    w = classifier_vector(None, None, X_train, y_train, alpha, None, gamma, 'ft', w_tilde)

    t1 = np.linspace(mean_c1 - 4*std, mean_c1 + 5*std, 100)
    t2 = np.linspace(mean_c2 - 4*std, mean_c2 + 5*std, 100)


    # Plot all
    ax[i].plot(t1, gaussian(t1, mean_c1, std), color = 'tab:red', linewidth= linewidth)
    ax[i].plot(t2, gaussian(t2, mean_c2, std), color = 'tab:blue', linewidth= linewidth)
    ax[i].set_xlabel('$\\mathbf{w}_{\\alpha}^\\top \\mathbf{x}$', fontsize = fontsize)

    # Plotting histogram
    ax[i].hist(X_test[:, (y_test < 0)].T @ w, color = 'tab:red', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].hist(X_test[:, (y_test > 0)].T @ w, color = 'tab:blue', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].tick_params(axis = 'x', which = 'both', labelsize = labelsize)
    ax[i].tick_params(axis = 'y', which = 'both', labelsize = labelsize)
    # Label: label = '$\mathcal{C}_2$'
    ax[i].set_title(f'$\\alpha = {alpha}$', fontsize = fontsize)

ax[0].set_ylabel(f'$ \| \\tilde w \| = {w_norm} $', fontsize = fontsize)

path = './study-plot' + f'/distribution_arbitrary-n-{n}-p-{p}-mu-{mu_beta}-w-{w_norm}-gamma-{gamma}.pdf'
fig.savefig(path, bbox_inches='tight')