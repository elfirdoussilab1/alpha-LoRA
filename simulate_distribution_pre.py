# This file is used to generate distribution plots on Synthetic data to validate theoretical calculus
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rmt_results import *
from dataset import *

plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})#,"font.sans-serif": "Helvetica",})

fix_seed(123)

# Parameters
p = 784
N = 5000
gamma = 1
type_1 = '1_7'
dataset_name = 'mnist'

# Datasets
data = MNIST(N, 6, 1, classifier= 'ft')
#train_data = LLM_dataset(N, 'sentiment_train', 'ft')
#test_data = LLM_dataset(N, 'sentiment_test', 'pre')

mu = data.mu
X_train, y_train = data.X_train.T, data.y_train
X_test, y_test = data.X_test.T, data.y_test

print(X_train.shape)
print(X_test.shape)
# Expectation of class C_1 and C_2
mean_c2 = test_expectation_pre(N, p, mu, gamma)
mean_c1 = - mean_c2
expec_2 = test_expectation_2_pre(N, p, mu, gamma)
std = np.sqrt(expec_2 - mean_c2**2)

# Classifier 
#print("Fine-tuning shape:", X_ft.shape)
w = classifier_vector(X_train, y_train, None, None, None, gamma, None, classifier= 'pre' )
t1 = np.linspace(mean_c1 - 5*std, mean_c1 + 5*std, 100)
t2 = np.linspace(mean_c2 - 5*std, mean_c2 + 5*std, 100)


# Plot all
plt.plot(t1, gaussian(t1, mean_c1, std), color = 'tab:red')
plt.plot(t2, gaussian(t2, mean_c2, std), color = 'tab:blue')
plt.xlabel('$\\mathbf{w}^\\top \\mathbf{x}$')

# Plotting histogram

plt.hist(X_test[:, (y_test < 0)].T @ w, color = 'tab:red', density = True, bins=25, alpha=.5, edgecolor = 'black')
plt.hist(X_test[:, (y_test > 0)].T @ w, color = 'tab:blue', density = True, bins=25, alpha=.5, edgecolor = 'black')
# Label: label = '$\mathcal{C}_2$'
plt.title(f'{dataset_name} {type_1[0]} and {type_1[2]} classifier')

plt.ylabel("Density")
path = './study-plot' + f'/distribution_real-{dataset_name}-N-{N}-p-{p}-mu-{round(mu, 3)}-gamma_pre-{gamma}.pdf'
plt.savefig(path, bbox_inches='tight')