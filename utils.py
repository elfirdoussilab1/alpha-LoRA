# Utile functions
import numpy as np
import random
import dataset
import json
import torch

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Data generation
def gaussian_mixture(n, vmu, pi=0.5, cov = False):
    p = len(vmu)
    y = np.ones(n)
    y[:int(n * pi)] = -1
    Z = np.random.randn(p, n) # (z_1, ..., z_n)
    # Adding a covariance matrix
    if cov : 
        M_2 = np.random.rand(p, p)
        C_2 = M_2.T @ M_2 / p
        Z[:, int(n * pi):] = C_2 @ Z[:, int(n * pi):]
    X = np.outer(vmu, y) + Z # np.outer = vmu.T @ y
    return X, y

def generate_data(N, n, p, mu, mu_orth, beta, classifier = 'ft'):  
    """
    Function to generate Pre-training synthetic data
    params:
        N (int): total number of Pre-training data vectors
        p (int): dimension of a single data vector
        pi (float): proportion of negative labels in pre-training data
        mu (float): norm of the mean of pre-training vectors
        mu_orth (float): norm of the orthogonal vector to vmu
    """
    vmu = np.zeros(p) # vecteur ligne: shape = (p,)
    vmu[0] = mu # vmu is of norm mu
    
    # Fine-tuning mean vector
    vmu_orth = np.zeros(p) # to be modified to the orthogonal vector
    vmu_orth[1] = mu_orth
    vmu_beta = beta * vmu + np.sqrt(1 - beta**2) * vmu_orth

    if 'ft' in classifier:
        return gaussian_mixture(n, vmu_beta), gaussian_mixture(20*n, vmu_beta)
    else: # pre-training data
        return gaussian_mixture(N, vmu), gaussian_mixture(20*N, vmu)

# Let us define classifier expression
def classifier_vector(X_pre, y_pre, X_ft, y_ft, alpha, gamma_pre, gamma_ft, classifier = 'ft', w_tilde = None):
    # Pre-training classifier
    if w_tilde is None:
        q_pre = np.linalg.solve(X_pre @ X_pre.T / X_pre.shape[1] + gamma_pre * np.eye(X_pre.shape[0]), np.eye(X_pre.shape[0]))
        w_pre = q_pre @ X_pre @ y_pre / X_pre.shape[1]
    else:
        w_pre = w_tilde

    if 'pre' in classifier:
        return w_pre
    
    # classifier a
    q_ft = np.linalg.solve(X_ft @ X_ft.T / X_ft.shape[1] + gamma_ft * np.eye(X_ft.shape[0]), np.eye(X_ft.shape[0]))
    a = q_ft @ (X_ft @ y_ft - alpha * X_ft @ X_ft.T @ w_pre) / X_ft.shape[1]

    return alpha * w_pre + a

# g(x) = <w, x>
g = lambda w, X: X.T @ w

# Labelling
decision = lambda w, X: 2 * (g(w, X) >= 0) - 1

# Binary accuracy function
def accuracy(y, y_pred):
    acc = np.mean(y == y_pred)
    return max(acc, 1 - acc)

# Losses
def L2_loss(w, X, y):
    # X of shape (p, n)
    return np.mean((X.T @ w - y)**2)

def empirical_accuracy(classifier, batch, N, n, p, mu, mu_orth, beta, alpha, gamma_pre, gamma_ft, data_type= 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            X_pre, y_pre = generate_data(N, n, p, mu, mu_orth, beta, 'pre')[0]
            X_ft, y_ft = generate_data(N, n, p, mu, mu_orth, beta, 'ft')[0]
            X_test, y_test = generate_data(N, n, p, mu, mu_orth, beta, classifier)[1]

        elif 'amazon' in data_type: # amazon_source_target or llm
            source = data_type.split('_')[1]
            target = data_type.split('_')[2]
            data_pre, data_ft, beta, vmu_orth = dataset.create_pre_ft_datasets(N, source, n, target, data_type)
            X_pre, y_pre = data_pre.X_train.T, data_pre.y_train
            X_ft, y_ft = data_ft.X_train.T, data_ft.y_train
            X_test, y_test = data_ft.X_test.T, data_ft.y_test
        elif 'mnist' in data_type: # mnist_clpre1_clpre2_clft1_clft2
            l = data_type.split('_')
            source = f'{l[1]}_{l[2]}'
            target = f'{l[3]}_{l[4]}'
            data_pre, data_ft, beta, vmu_orth = dataset.create_pre_ft_datasets(N, source, n, target, data_type)
            X_pre, y_pre = data_pre.X_train.T, data_pre.y_train
            X_ft, y_ft = data_ft.X_train.T, data_ft.y_train
            X_test, y_test = data_ft.X_test.T, data_ft.y_test

        else: # LLM
            print("Not yet!")
        w = classifier_vector(X_pre, y_pre, X_ft, y_ft, alpha, gamma_pre, gamma_ft, classifier)

        res += accuracy(y_test, decision(w, X_test))
    return res / batch

def empirical_mean(classifier, batch, N, n, p, mu, mu_orth, beta, alpha, gamma_pre, gamma_ft, data_type= 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            X_pre, y_pre = generate_data(N, n, p, mu, mu_orth, beta, 'pre')[0]
            X_ft, y_ft = generate_data(N, n, p, mu, mu_orth, beta, 'ft')[0]
            X_test, y_test = generate_data(N, n, p, mu, mu_orth, beta, classifier)[1]

        else: # real data
            return -1
        w = classifier_vector(X_pre, y_pre, X_ft, y_ft, alpha, gamma_pre, gamma_ft, classifier)

        res += np.mean(y_test * (X_test.T @ w))
    return res / batch

def empirical_mean_2(classifier, batch, N, n, p, mu, mu_orth, beta, alpha, gamma_pre, gamma_ft, data_type= 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            X_pre, y_pre = generate_data(N, n, p, mu, mu_orth, beta, 'pre')[0]
            X_ft, y_ft = generate_data(N, n, p, mu, mu_orth, beta, 'ft')[0]
            X_test, y_test = generate_data(N, n, p, mu, mu_orth, beta, classifier)[1]

        else: # real data
            return -1
        w = classifier_vector(X_pre, y_pre, X_ft, y_ft, alpha, gamma_pre, gamma_ft, classifier)

        res += np.mean((X_test.T @ w)**2)
    return res / batch

def empirical_risk(classifier, batch, N, n, p, mu, mu_orth, beta, alpha, gamma_pre, gamma_ft, data_type= 'synthetic'):
    res = 0
    for i in range(batch):
        if 'synthetic' in data_type:
            X_pre, y_pre = generate_data(N, n, p, mu, mu_orth, beta, 'pre')[0]
            X_ft, y_ft = generate_data(N, n, p, mu, mu_orth, beta, 'ft')[0]
            X_test, y_test = generate_data(N, n, p, mu, mu_orth, beta, classifier)[1]

        else: # real data
            return -1
        w = classifier_vector(X_pre, y_pre, X_ft, y_ft, alpha, gamma_pre, gamma_ft, classifier)

        res += L2_loss(w, X_test, y_test)
    return res / batch


###------------------------ Functions for arbitrary classifier -----------------------
def empirical_accuracy_arbitrary(batch, n, p, w_tilde, vmu_beta, alpha, gamma, data_type = 'synthetic'):
    # n is the number of target data
    res = 0
    assert len(vmu_beta) == p
    for i in range(batch):
        if 'synthetic' in data_type:
            X_train, y_train = gaussian_mixture(n, vmu_beta)
            X_test, y_test = gaussian_mixture(20*n, vmu_beta)
        elif 'amazon' in data_type: # amazon_target
            target = data_type.split('_')[1]
            data = dataset.Amazon(n, target, 'pre')
            X_test, y_test = data.X_test.T, data.y_test
            X_train, y_train = data.X_train.T, data.y_train
        else:
            print("Not implemented yet !")
        
        w = classifier_vector(None, None, X_train, y_train, alpha, None, gamma, 'ft', w_tilde)
        res += accuracy(y_test, decision(w_tilde, X_test))
    return res / batch

def empirical_mean_arbitrary(batch, n, p, w_tilde, vmu_beta, alpha, gamma, data_type = 'synthetic'):
    # n is the number of target data
    res = 0
    assert len(vmu_beta) == p
    for i in range(batch):
        if 'synthetic' in data_type:
            X_train, y_train = gaussian_mixture(n, vmu_beta)
            X_test, y_test = gaussian_mixture(20*n, vmu_beta)
        elif 'amazon' in data_type: # amazon_target
            target = data_type.split('_')[1]
            data = dataset.Amazon(n, target, 'ft')
            X_test, y_test = data.X_test.T, data.y_test
            X_train, y_train = data.X_train.T, data.y_train
        else:
            print("Not implemented yet !")
        
        w = classifier_vector(None, None, X_train, y_train, alpha, None, gamma, 'ft', w_tilde)
        res += np.mean(y_test * (X_test.T @ w))
    return res / batch

def empirical_mean_2_arbitrary(batch, n, p, w_tilde, vmu_beta, alpha, gamma, data_type = 'synthetic'):
    # n is the number of target data
    res = 0
    assert len(vmu_beta) == p
    for i in range(batch):
        if 'synthetic' in data_type:
            X_train, y_train = gaussian_mixture(n, vmu_beta)
            X_test, y_test = gaussian_mixture(20*n, vmu_beta)
        elif 'amazon' in data_type: # amazon_target # Not tested yet
            target = data_type.split('_')[1]
            data = dataset.Amazon(n, target, 'ft')
            X_test, y_test = data.X_test.T, data.y_test
            X_train, y_train = data.X_train.T, data.y_train
        else:
            print("Not implemented yet !")
        
        w = classifier_vector(None, None, X_train, y_train, alpha, None, gamma, 'ft', w_tilde)
        res += np.mean((X_test.T @ w)**2)
    return res / batch

def empirical_risk_arbitrary(batch, n, p, w_tilde, vmu_beta, alpha, gamma, data_type = 'synthetic'):
    # n is the number of target data
    res = 0
    assert len(vmu_beta) == p
    for i in range(batch):
        if 'synthetic' in data_type:
            X_train, y_train = gaussian_mixture(n, vmu_beta)
            X_test, y_test = gaussian_mixture(20*n, vmu_beta)
        elif 'amazon' in data_type: # amazon_target
            target = data_type.split('_')[1]
            data = dataset.Amazon(n, target, 'ft')
            X_test, y_test = data.X_test.T, data.y_test
            X_train, y_train = data.X_train.T, data.y_train
        else:
            print("Not implemented yet !")
        
        w = classifier_vector(None, None, X_train, y_train, alpha, None, gamma, 'ft', w_tilde)
        res += L2_loss(w, X_test, y_test)
    return res / batch   

# Gaussian density function
def gaussian(x, mean, std):
    return np.exp(- (x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

# Evaluating BERT models
@torch.no_grad
def evaluate_bert_accuracy(model, loader, device = 'cuda'):
    model.eval()
    total_correct = 0
    total_samples = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Perform a forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Calculate accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0) # Add the number of samples in the current batch
    
    # Calculate accuracy over the entire dataset
    return total_correct / total_samples