# This file is used to generate distribution plots on Synthetic data to validate theoretical calculus
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *
from dataset import *

#plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

fix_seed(123)

# Parameters
'''
p = 400
N = 2000
n = 50
gamma_pre = 1
gamma_ft = 1
type_1 = 'book'
type_2 = 'dvd'
'''
p = 1000
N = 20000
n = 100
gamma_pre = 1
gamma_ft = 1
type_1 = 'sentiment_train'
type_2 = 'sentiment_test'

# Datasets
data_pre, data_ft, beta, vmu_orth = create_pre_ft_datasets(N, type_1, n, type_2, dataset_name= 'llm')
mu_orth = np.linalg.norm(vmu_orth)
mu = data_pre.mu
X_pre, y_pre = data_pre.X_train.T, data_pre.y_train
X_ft, y_ft = data_ft.X_train.T, data_ft.y_train

#alpha = optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft)[0]
alphas = [0.1, 1, 10]

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
    #print("Fine-tuning shape:", X_ft.shape)
    w = classifier_vector(X_pre, y_pre, X_ft, y_ft, alpha, gamma_pre, gamma_ft)
    t1 = np.linspace(mean_c1 - 4*std, mean_c1 + 5*std, 100)
    t2 = np.linspace(mean_c2 - 4*std, mean_c2 + 5*std, 100)


    # Plot all
    ax[i].plot(t1, gaussian(t1, mean_c1, std), color = 'tab:red', linewidth= linewidth)
    ax[i].plot(t2, gaussian(t2, mean_c2, std), color = 'tab:blue', linewidth= linewidth)
    ax[i].set_xlabel('$\\mathbf{w}_{\\alpha}^\\top \\mathbf{x}$', fontsize = fontsize)

    # Plotting histogram
    X_test, y_test = data_ft.X_test.T, data_ft.y_test
    ax[i].hist(X_test[:, (y_test < 0)].T @ w, color = 'tab:red', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].hist(X_test[:, (y_test > 0)].T @ w, color = 'tab:blue', density = True, bins=25, alpha=.5, edgecolor = 'black')
    ax[i].tick_params(axis = 'x', which = 'both', labelsize = labelsize)
    ax[i].tick_params(axis = 'y', which = 'both', labelsize = labelsize)
    # Label: label = '$\mathcal{C}_2$'
    ax[i].set_title(f'$\\alpha = {alpha}$', fontsize = fontsize)

ax[0].set_ylabel(f"{type_1.capitalize()} To {type_2.capitalize()}", fontsize = fontsize)
path = './study-plot' + f'/distribution_real-{type_1}-{type_2}-N-{N}-n-{n}-p-{p}-alpha-{alpha}-mu-{round(mu, 2)}-mu_orth-{round(mu_orth, 2)}-beta-{round(beta, 2)}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}.pdf'
fig.savefig(path, bbox_inches='tight')