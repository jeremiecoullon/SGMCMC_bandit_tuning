import numpy as np
import time
import os
from pathlib import Path

import jax.numpy as jnp
from jax import random, jit, grad, vmap, partial

import context
from examples.logistic_regression.logistic_regression_model import gen_data_and_log_post
from samplers.sgld import sgld_jaxloop, sgld_kernel, sgld_time_budget
from samplers.sgldcv import sgld_kernel_CV, sgld_time_budget_CV
from ksd import imq_KSD

"""
Feb 2021
Grid search for logistic regression. Uses a time budget
"""


dim = 10
Ndata = 100000
key = random.PRNGKey(0) # same key as 'LR-evaluate_default_vs_KSD.ipynb'
theta_true, X, y_data, val_and_grad_log_post = gen_data_and_log_post(key, dim, Ndata)

# testing and training data
num_train = int(Ndata*0.8)

X_train = X[:num_train]
X_test = X[num_train:]

y_train = y_data[:num_train]
y_test = y_data[num_train:]
del X, y_data


def LR_grad(theta):
    _, mygrad = val_and_grad_log_post(theta, X_train, y_train)
    return mygrad

batch_LR_grad = jit(vmap(LR_grad))


def get_ksd_LR_2_KSD(key, log_dt, log_batch_size_ratio, time_budget, CV):
    dt = 10**log_dt
    batch_size_ratio = 10**(log_batch_size_ratio)
    batch_size = int(X_train.shape[0]*batch_size_ratio)
    thin_step = 2
    if CV == False:
        sgld_samples, sgld_grads = sgld_time_budget(key, val_and_grad_log_post, X_train,
                                                    y_train, time_budget, dt,
                                                    batch_size=batch_size, x_0=theta_true)
    else:
        sgld_samples, sgld_grads = sgld_time_budget_CV(key, val_and_grad_log_post, X_train,
                                                       y_train, time_budget, dt,
                                                       batch_size=batch_size, mean_param=theta_true, x_0=theta_true)
    est_KSD_true_gradients = imq_KSD(sgld_samples[::thin_step], batch_LR_grad(sgld_samples[::thin_step]))
    return est_KSD_true_gradients


def grid_search_2D_LR(time_budget, logdt_grid, log_b_s_r_grid, CV):
    key = random.PRNGKey(100)
    XX, YY = np.meshgrid(logdt_grid, log_b_s_r_grid)
    ksd_mat_LR = np.zeros((len(log_b_s_r_grid), len(logdt_grid)))

    for i in range(len(log_b_s_r_grid)):
        for j in range(len(logdt_grid)):
            print(f"trying out {i,j}..")
            ksd_mat_LR[i,j] = get_ksd_LR_2_KSD(key, logdt_grid[j], log_b_s_r_grid[i], time_budget, CV)
            print(f"KSD for (logdt, log_b_s_r)={logdt_grid[j], log_b_s_r_grid[i]}: {ksd_mat_LR[i,j]:.2f}")

    return ksd_mat_LR



    
# bool_list = [True, False]
# time_budget_list = [5.]



bool_list = [False]
time_budget_list = [4.]


for CV_bool in bool_list:
    if CV_bool:
        # high resolution version
        logdt_grid = np.linspace(-5.2,-3, 22)
        log_b_s_r_grid = np.linspace(-4, 0.5, 17)
    else:
        # high resolution version
        logdt_grid = np.linspace(-5.3,-3.6, 22)
        log_b_s_r_grid = np.linspace(-2.2,0, 17)
    for time_budget in time_budget_list:
        CV_string = "CV-" if CV_bool else ""

        print(f"\nRunning sgld{CV_string[:-1]} for a time budget of {time_budget}sec..")

        output_dir = f"outputs/LR_2d_grid_search/sgld_{CV_string}{str(time_budget)}sec"
        Path(output_dir).mkdir(exist_ok=True)
        ksd_mat_LR = grid_search_2D_LR(time_budget, logdt_grid, log_b_s_r_grid, CV_bool)
        np.savetxt(f"{output_dir}/LR_2d_grid_search{CV_string}{str(time_budget)}sec.txt", ksd_mat_LR)
        print("Done.")
