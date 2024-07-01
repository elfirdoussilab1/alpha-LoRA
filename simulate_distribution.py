# This file is used to generate distribution plots on Synthetic data to validate theoretical calculus
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})


# Parameters
p = 400
n = 200
N = 5000
mu = 1.5
mu_orth = 1
beta = 0.8
gamma_pre = 1
gamma_ft = 1

# Datasets
X_pre, y_pre = generate_data(N, n, p, mu, mu_orth, beta, 'pre')[0]
(X_ft, y_ft), (X_test, y_test) = generate_data(N, n, p, mu, mu_orth, beta, 'ft')

alphas = [1e-1, 1, 10]
fig, ax = plt.subplots(1, 3, figsize = (30, 6))
fontsize = 40
labelsize = 35
linewidth = 3

for i, alpha in enumerate(alphas):

    # Expectation of class C_1 and C_2
    mean_c2 = test_expectation(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft)
    mean_c1 = - mean_c2
    expec_2 = test_expectation_2(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft)
    std = np.sqrt(expec_2 - mean_c2**2)

    # Classifier 
    w = classifier_vector(X_pre, y_pre, X_ft, y_ft, alpha, gamma_pre, gamma_ft)
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

ax[0].set_ylabel(f'$ \\beta = {beta} $', fontsize = fontsize)

path = './study-plot' + f'/distribution-N-{N}-n-{n}-p-{p}-mu-{mu}-mu_orth-{mu_orth}-beta-{beta}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}.pdf'
fig.savefig(path, bbox_inches='tight')