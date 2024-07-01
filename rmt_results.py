# In this file, we will implement all the theoretical results gotten using RMT
import numpy as np
from utils import *
import scipy.integrate as integrate
import utils

def Delta(eta, gamma):
    return (eta - gamma - 1 + np.sqrt((eta - gamma - 1)**2 + 4*eta*gamma)) / (2 * gamma)

def resolvent(vmu, delta, gamma):
    p = len(vmu)
    mu = np.sqrt(np.sum(vmu**2))
    r = (1 + delta) / (1 + gamma * (1 + delta))
    M = np.eye(p) - np.outer(vmu, vmu) / (mu**2 + 1 + gamma * (1 + delta))
    return r * M

def test_expectation(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft):

    delta_R = Delta(p/N, gamma_pre)
    delta_Q = Delta(p/n, gamma_ft)
    mu_beta = np.sqrt((beta * mu)**2 + (1 - beta**2) * mu_orth**2)
    r = 1 / (mu_beta**2 + 1 + gamma_ft * (1 + delta_Q))
    s = mu_beta**2 + alpha * gamma_ft * (1 + delta_Q) * beta * mu**2 / (mu**2 + 1 + gamma_pre * (1 + delta_R))
    return r * s

def denom(gamma, p, n):
    eta = p/n
    delta = Delta(eta, gamma)
    return 1 - eta / (1 + gamma * (1 + delta))**2 

def test_expectation_2(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft):

    eta = p/n
    eta_tilde = p / N
    # mu_beta**2
    mu_beta_2 = (beta * mu)**2 + (1 - beta**2) * mu_orth**2
    
    # Needed quantities
    h = denom(gamma_ft, p, n)
    h_tilde = denom(gamma_pre, p, N)
    delta_R = Delta(eta_tilde, gamma_pre)
    delta_Q = Delta(eta, gamma_ft)
    lambda_R = mu**2 + 1 + gamma_pre * (1 + delta_R)
    lambda_Q = mu_beta_2 + 1 + gamma_ft * (1 + delta_Q)
    
    # T_1
    r_1 = mu_beta_2 / (h * lambda_Q)
    T_1 = r_1 * ((mu_beta_2 + 1) / lambda_Q - 2 * (1 - h)) + (1 - h) / h

    # T_2
    r_2 = 2 * gamma_ft * beta * (1 + delta_Q) * mu**2 / (lambda_R * lambda_Q)
    T_2 = r_2 * (1 - gamma_ft * (1 + delta_Q) / (h * lambda_Q))

    # T_3
    r_3 = (gamma_ft * (1 + delta_Q))**2 / h
    T_3 = r_3 * ( (mu**2 / lambda_R**2) * ((beta*mu)**2 / lambda_Q**2 + ((1 - h) / eta) * (1 + (beta*mu)**2 * mu_beta_2 / lambda_Q**2 - 2*(beta*mu)**2 / lambda_Q)))
    T_3 += r_3 * ((1 - h) * (1 - h_tilde) / eta) * (1 - 2 * mu**2 / lambda_R)
    return T_1 + alpha * T_2 + (alpha**2) * T_3

def test_accuracy(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft):

    # E[g] and E[g^2]
    mean = test_expectation(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft)
    expec_2 = test_expectation_2(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft)
    std = np.sqrt(expec_2 - mean**2)
    return 1 - integrate.quad(lambda x: utils.gaussian(x, 0, 1), abs(mean)/std, np.inf)[0]

def test_risk(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft):
    # E[g] and E(g^2)
    mean = test_expectation(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft)
    expec_2 = test_expectation_2(N, n, p, mu, mu_orth, alpha, beta, gamma_pre, gamma_ft)
    return expec_2 + 1 - 2 * mean

# Minimum and Maximum alphas
def optimal_alphas(N, n, p, mu, mu_orth, beta, gamma_pre, gamma_ft):
    eta = p/n
    mu_beta_2 = (beta * mu)**2 + (1 - beta**2) * mu_orth**2
    
    # Needed quantities
    h = denom(gamma_ft, p, n)
    h_tilde = denom(gamma_pre, p, N)
    delta_R = Delta(p/N, gamma_pre)
    delta_Q = Delta(p/n, gamma_ft)
    lambda_R = mu**2 + 1 + gamma_pre * (1 + delta_R)
    lambda_Q = mu_beta_2 + 1 + gamma_ft * (1 + delta_Q)
    
    # T_1
    r_1 = mu_beta_2 / (h * lambda_Q)
    T_1 = r_1 * ((mu_beta_2 + 1) / lambda_Q - 2 * (1 - h)) + (1 - h) / h

    # T_2
    r_2 = 2 * gamma_ft * beta * (1 + delta_Q) * mu**2 / (lambda_R * lambda_Q)
    T_2 = r_2 * (1 - gamma_ft * (1 + delta_Q) / (h * lambda_Q))

    # T_3
    r_3 = (gamma_ft * (1 + delta_Q))**2 / h
    T_3 = r_3 * ( (mu**2 / lambda_R**2) * ((beta*mu)**2 / lambda_Q**2 + ((1 - h) / eta) * (1 + (beta*mu)**2 * mu_beta_2 / lambda_Q**2 - 2*(beta*mu)**2 / lambda_Q)))
    T_3 += r_3 * ((1 - h) * (1 - h_tilde) / eta) * (1 - 2 * mu**2 / lambda_R)

    # Max alpha
    alpha_max = (-2*T_1*beta*delta_Q*gamma_ft*mu**2 - 2*T_1*beta*gamma_ft*mu**2 + T_2*lambda_R*mu_beta_2)
    alpha_max = alpha_max / (T_2*beta*delta_Q*gamma_ft*mu**2 + T_2*beta*gamma_ft*mu**2 - 2*T_3*lambda_R*mu_beta_2)

    # Min alpha
    alpha_min = - lambda_R * mu_beta_2 / (beta * gamma_ft * mu**2 * (1 + delta_Q))

    return alpha_max, alpha_min