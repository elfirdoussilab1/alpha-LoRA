from utils import *
from rmt_results import *
import pandas as pd
from tqdm.auto import tqdm
from dataset import *

# Parameters for MNIST
N = 1000
p = 784
n = 10
gamma_pre = 1
gamma_ft = 1
batch = 1

#data_names = [f'{a}_{b}' for a in range(10) for b in range(10) if a != b]
#data_source_target = list({('6_9', b) for b in data_names})
data_source_target = [('6_9', '6_7'), ('6_9', '0_7')]
results = pd.DataFrame(columns=['Target', 'Optimal-alpha', 'No-FT', 'std-no-ft', 'LoRA', 'std-LoRA', 'Optimal','std_optimal'])
dataset_name = 'mnist'

alphas = np.linspace(-2, 2, 21)
alpha_zero_idx = np.where(alphas == 0)[0][0]
alpha_one_idx = np.where(alphas == 1)[0][0]
seeds = [1, 123, 404]

for source, target in tqdm(data_source_target):
    data_type = 'mnist_' + source + '_' + target
    accs_means = []
    accs_std = []
    for alpha in alphas:
        acc = []
        for seed in seeds:
            fix_seed(seed)
            acc.append(empirical_accuracy('ft', batch, N, n, p, None, None, None, alpha, gamma_pre, gamma_ft, data_type))
        accs_means.append(round(np.mean(acc) * 100, 2))
        accs_std.append(round(np.std(acc) * 100, 2))
    alpha_max_idx = np.argmax(accs_means)

    row = {'Target': target,
            'Optimal-alpha': round(alphas[alpha_max_idx], 3),
            'No-FT': accs_means[alpha_zero_idx],
           'std-no-ft' : accs_std[alpha_zero_idx],
           'LoRA': accs_means[alpha_one_idx],
           'std-LoRA': accs_std[alpha_one_idx],
           'Optimal' : accs_means[alpha_max_idx],
           'std_optimal' : accs_std[alpha_max_idx]
           }
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

path = './results-data/' + f'accuracy_comp_mnist-N-{N}-n-{n}-p-{p}-gamma_pre-{gamma_pre}-gamma_ft-{gamma_ft}-batch-{batch}.csv'
results.to_csv(path)
