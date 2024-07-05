# This file will be used to generate the values of accuracies in the table 1 in the paper.
from utils import *
from rmt_results import *
import pandas as pd
from tqdm.auto import tqdm
from dataset import *

# Parameters
'''
N = 2000
p = 400
n = 20
gamma_pre = 1
gamma_ft = 1
batch = 10
'''
N = 40000
p = 1000
n = 200
gamma_pre = 1e-1
gamma_ft = 1e-1
batch = 10

# Create results in a dataframe: rows = datasets (4), columns = algorithms (3)
dataset_name = 'llm'
#data_names = ['book', 'dvd', 'elec', 'kitchen']
data_names = ['sentiment', 'safety']
#data_source_target = list({(a, b) for a in data_names for b in data_names if a != b})
data_source_target = [('safety', 'sentiment')]
results = pd.DataFrame(columns=['Optimal-alpha', 'No-FT', 'std-no-ft', 'Optimal','std_optimal'])
seeds = [1, 123, 404]

for source, target in data_source_target:
    #data_type = 'amazon_' + source + '_' + target
    data_type = 'llm_' + source + '_' + target

    # Datasets
    data_pre, data_ft, beta, vmu_orth = dataset.create_pre_ft_datasets(N, source, n, target, dataset_name= dataset_name)
    mu_orth = np.linalg.norm(vmu_orth)
    mu = data_pre.mu
    alpha_opt = optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft)[0]

    acc_optimal = []
    acc_noft = []

    for seed in tqdm(seeds):
        fix_seed(seed)
        # Optimal
        acc_optimal.append(empirical_accuracy('ft', batch, N, n, p, mu, mu_orth, beta, alpha_opt, gamma_pre, gamma_ft, data_type))

        # No-FT
        acc_noft.append(empirical_accuracy('ft', batch, N, n, p, mu, mu_orth, beta, 0, gamma_pre, gamma_ft, data_type))

    row = {'Optimal-alpha': round(alpha_opt, 2),
            'No-FT': round(np.mean(acc_noft) * 100, 2),
           'std-no-ft' : round(np.std(acc_noft) * 100, 2),
           'Optimal' : round(np.mean(acc_optimal) * 100, 2),
           'std_optimal' : round(np.std(acc_optimal) * 100, 2)
           }
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

results.index = data_source_target
path = './results-data/' + f'accuracy_comp-N-{N}-n-{n}-p-{p}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}-batch-{batch}.csv'
results.to_csv(path)
