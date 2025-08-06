# In this file, we will verify our theoretical results on fine-tuning regression tasks on real world datasets.
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm
from utils import *
from dataset import LinearRegressionDatasetLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a DistilBERT model with Adapters on GLUE task")

    # Training arguments
    parser.add_argument("--n", type=int, default=None, help="The number of training samples")
    parser.add_argument("--gamma", type=float, default=1, help="Weight decay")
    parser.add_argument("--sigma", type=float, default=0.1, help="Additive noise")
    parser.add_argument("--whiten", type=lambda x: x.lower() == 'true', default=None, help="Boolean to choose if we apply whitening")
    parser.add_argument("--weighting", type=lambda x: x.lower() == 'true', default=None, help="Boolean to choose if we apply the weights to estimate alpha")

    args = parser.parse_args()
    return args

args = parse_args()
# Hyperparameters
p = 8
d = 1

# The purpose of this file is to make a table of results just like the Amazon review experiments.
# We want to make plots about the test risk evolution with alpha, and hopefully see that our estimated alpha^* is the optimal. 
# If we get this, we finish the experiments about this part.
#data_names = ['energy', 'wine', 'california', 'diabetes']
data_names = ['energy', 'california', 'diabetes']
data_source_target = [(a, b) for a in data_names for b in data_names if a != b]
results = pd.DataFrame(columns=['Optimal-alpha', 'No-FT', 'std-no-ft', 'LoRA', 'std-LoRA', 'Optimal','std_optimal'])

seeds = [1, 123, 404]
for source, target in data_source_target:
    # Get datasets used to estimate alpha_opt
    data_source = LinearRegressionDatasetLoader(source, n = None, p=p, d=d,  sigma = args.sigma, whiten = args.whiten)
    data_target = LinearRegressionDatasetLoader(target, n = None, p=p, d=d, sigma = args.sigma, whiten = args.whiten)

    X_s, Y_s = data_source.get_data()
    X_t, Y_t = data_target.get_data()

    # Estimate optimal alpha
    #alpha_opt = estimate_alpha_reg(X_t, Y_t, X_s, Y_s)

    # Estimate W_s
    W_s = estimate_reg_matrix(X_s.T, Y_s.T, args.weighting)
    W_t = estimate_reg_matrix(X_t.T, Y_t.T, args.weighting)
    alpha_opt = np.trace(W_s @ W_t.T) / np.trace(W_s @ W_s.T)

    assert W_s.shape == (d, p)

    # Compute risks
    risk_optimal = []
    risk_noft = []
    risk_lora = []

    for seed in tqdm(seeds):
        fix_seed(seed)
        data_target = LinearRegressionDatasetLoader(target, n = args.n, p=p, d=d, sigma = args.sigma, whiten = args.whiten, random_state=seed)
        X_train, X_test, Y_train, Y_test = data_target.get_data()

        # Optimal
        risk_optimal.append(empirical_risk_real(X_train, Y_train, X_test, Y_test, W_s, alpha_opt, args.gamma))

        # No-FT
        risk_noft.append(empirical_risk_real(X_train, Y_train, X_test, Y_test, W_s, 0, args.gamma))

        # LoRA
        risk_lora.append(empirical_risk_real(X_train, Y_train, X_test, Y_test, W_s, 1, args.gamma))

    row = {
            'Optimal-alpha': alpha_opt,
            'No-FT': np.mean(risk_noft),
            'std-no-ft' : np.std(risk_noft),
            'LoRA': np.mean(risk_lora),
            'std-LoRA': np.std(risk_lora),
            'Optimal' : np.mean(risk_optimal),
            'std_optimal' :np.std(risk_optimal)
           }
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

results.index = data_source_target
path = './results-data/' + f'risk_regression_comp-n-{args.n}-sigma-{args.sigma}-gamma-{args.gamma}-whiten-{args.whiten}-weighting-{args.weighting}.csv'
results.to_csv(path)