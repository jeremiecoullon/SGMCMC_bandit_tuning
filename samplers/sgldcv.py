# -*- coding: utf-8 -*-

import numpy as np
import jax.numpy as jnp
from jax import random, partial, jit, lax
import time
from util import progress_bar_scan, wait_until_computed

@partial(jit, static_argnums=(3,7,8,9))
def sgld_kernel_CV(key, param, param_grad, log_post, dt, X, y_data, N_data, dim, batch_size, mean_param, grad_at_mean):
    subkey1, subkey2 = random.split(key)
    param = param + (dt/2)*param_grad + jnp.sqrt(dt)*random.normal(key=subkey1, shape=(dim,))
    idx_batch = random.choice(key=subkey2, a=jnp.arange(N_data), shape=(batch_size,))
    _, batch_param = log_post(param, X[idx_batch,:], y_data[idx_batch])
    _, batch_mean = log_post(mean_param, X[idx_batch, :], y_data[idx_batch])
    param_grad = grad_at_mean + batch_param - batch_mean
    return param, param_grad

@partial(jit, static_argnums=(1,4,6))
def sgld_jaxloop_CV(key, log_post, X, y_data, Nsamples, dt, batch_size, x_0, mean_param):
    N_data, dim = X.shape
    grad_at_mean = log_post(mean_param, X, y_data)[1]

    @progress_bar_scan(Nsamples)
    def body(carry, x):
        key, param, param_grad = carry
        key, subkey = random.split(key)
        param, param_grad = sgld_kernel_CV(subkey, param, param_grad,
                            log_post, dt, X, y_data, N_data, dim, batch_size,
                                           mean_param, grad_at_mean)
        return (key, param, param_grad), (param, param_grad)

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    _, param_grad = log_post(x_0, X[idx_batch, :], y_data[idx_batch])
    carry = (key, x_0, param_grad)
    carry, (samples, grads) = lax.scan(body, carry, jnp.arange(Nsamples))
    return samples, grads



def sgld_time_budget_CV(key, log_post, X, y_data, time_budget, dt, batch_size,
                         mean_param, x_0, verbose=0, ):
    N_data, dim = np.shape(X)
    samples = []#np.zeros((Nsamples, dim))
    grads = []#np.zeros((Nsamples, dim))
    X = jnp.array(X)
    y_data = jnp.array(y_data) # convert to jax array to avoid bug when doing `y_data[idx_batch]`
    param = x_0

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    _, param_grad = log_post(param, X[idx_batch,:], y_data[idx_batch])
    grad_at_mean = log_post(mean_param, X, y_data)[1]
    # run it once to compile the function
    wait_until_computed(sgld_kernel_CV(subkey, param, param_grad, log_post,
                                    dt, X, y_data, N_data, dim, batch_size, mean_param, grad_at_mean))

    start_time = time.time()
    if verbose==1:
        print(f"Running SGLD for a budget of {time_budget} seconds with batch size {batch_size}")
    while time.time() - start_time < time_budget:
        key, subkey = random.split(key)
        param, param_grad = sgld_kernel_CV(subkey, param, param_grad, log_post,
                        dt, X, y_data, N_data, dim, batch_size, mean_param, grad_at_mean)
        samples.append(param)
        grads.append(param_grad)

    end_time = time.time()
    if verbose==1:
        print("Done.")
        print(f"Running time: {(end_time-start_time):.1f} sec")
        print(f"Generated {np.shape(samples)[0]} samples")

    return np.array(samples), np.array(grads)
