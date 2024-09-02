import pandas as pd
import numpy as np
import os
import time
from sklearn.utils.extmath import randomized_svd
import random
import sys

def afMF(X, iteration=int(1e4), tolerence=1e-4, lambda_P=0, lambda_Q=0, sigma=3, random_seed=42):
    
    colnames = X.columns
    rownames = X.index
    
    X = np.array(X)

    log_norm_X = log_norm(X)

    k = k_rank_selection(log_norm_X, sigma, random_seed)
    print('k=',k)
    num_latent_features = k
    alpha = 0

    I = X.shape[0]
    J = X.shape[1]

    imputed_X, convergence = imputation_zero_alpha_regularization(log_norm_X, I, J, lambda_P, lambda_Q,
                                                                 num_latent_features, iteration, tolerence, random_seed)

    threshold_c = np.percentile(imputed_X, 0.001, axis=1)
    for i in range(imputed_X.shape[0]):
        imputed_X[i, :] = np.where(imputed_X[i, :] < threshold_c[i], 0, imputed_X[i, :])

    imputed_X_replace_zero = np.where(imputed_X == 0, np.nan, imputed_X)
    X_replace_zero = np.where(log_norm_X == 0, np.nan, log_norm_X)

    imputed_mean = np.nanmean(imputed_X_replace_zero, axis=1)
    imputed_std = np.nanstd(imputed_X_replace_zero, axis=1)
    original_mean = np.nanmean(X_replace_zero, axis=1)
    original_std = np.nanstd(X_replace_zero, axis=1)

    for i in range(imputed_X.shape[0]):
        imputed_X[i, :] = np.where(imputed_X[i, :] == 0, 0,
                                   (imputed_X[i, :] - imputed_mean[i]) * original_std[i] / imputed_std[i] + \
                                   original_mean[i])

    # imputed_X[np.logical_and(imputed_X == 0, X != 0)] = X[np.logical_and(imputed_X == 0, X != 0)]
    imputed_X[np.logical_and(imputed_X == 0, log_norm_X != 0)] = log_norm_X[
        np.logical_and(imputed_X == 0, log_norm_X != 0)]

    imputed_data_df = pd.DataFrame(imputed_X)
    imputed_data_df.columns = colnames
    imputed_data_df.index = rownames

    return imputed_data_df

def log_norm(X):
    sum_X = np.sum(X, axis = 0)
    output_X = np.log(X/sum_X * 10000 + 1)
    return output_X


def k_rank_selection(X, sigma=3, random_seed=42):
    if X.shape[1] < 100:
        print('number of features < 100')

    #U, s, Vh = randomized_svd(X.T, n_components=X.shape[0], n_iter = 2,  random_state=0)
    U, s, Vh = randomized_svd(X.T, n_components=100, n_iter = 2,  random_state=random_seed)
    spacing = s[:-1] - s[1:]

    start_pt = 78
    end_pt = 99
    #end_pt = 98

    spacing_mean = np.mean(spacing[start_pt:end_pt])
    spacing_std = np.std(spacing[start_pt:end_pt])

    k = np.max(np.where(spacing > (spacing_mean + sigma * spacing_std), np.arange(len(spacing)), 0))

    return k

def imputation_zero_alpha_regularization(X, I, J, lambda_P, lambda_Q, num_latent_features, iteration, tolerence, random_seed=42):
    stopped = False

    # initialize P, Q
    np.random.seed(random_seed)
    P = np.random.rand(I, num_latent_features)
    Q = np.random.rand(J, num_latent_features)
    #Q = np.random.rand(J, num_latent_features, random_state=42)

    # SVD with regularization
    old_error = 1e10
    old_obj = 1e10
    for k in range(iteration):

        #print('iteration: ', k)

        old_P = P.copy()
        old_Q = Q.copy()

        P = (X @ Q) @ np.linalg.inv(Q.T @ Q - lambda_P * np.identity(num_latent_features))

        Q = (X.T @ P) @ np.linalg.inv(P.T @ P - lambda_Q * np.identity(num_latent_features))

        error = np.sum((old_P - P) ** 2) + np.sum((old_Q - Q) ** 2)
        obj = np.sum((np.matmul(P, Q.T) - X) ** 2) + lambda_P*np.sum(P**2) + lambda_Q*np.sum(Q**2)

        ratio = np.sum((np.matmul(P, Q.T) - X) ** 2) / (np.sum(P**2) + np.sum(Q**2))

        #print('Objective Decreasing: ', (obj < old_obj))
        old_obj = obj
        #print('Model Error Decreasing: ', ((np.sqrt(np.sum((np.matmul(P, Q.T) - X) ** 2))/(I*J)) < old_error))
        old_error = (np.sqrt(np.sum((np.matmul(P, Q.T) - X) ** 2))/(I*J))
        #print('Model Error: ', old_error)
        #print('Serial Error: ', error)
        #print('Loss to Norm Ratio:', ratio)

        if error < tolerence:
            stopped = True
            break

    if stopped:
        print("converge")
    else:
        print("not converge")

    # imputation
    output = np.matmul(P, Q.T)

    return output, stopped
