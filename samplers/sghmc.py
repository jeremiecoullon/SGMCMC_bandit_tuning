# -*- coding: utf-8 -*-

import numpy as np
import jax.numpy as jnp
from jax import random, partial, jit, lax
import time
from util import progress_bar_scan, wait_until_computed

@partial(jit, static_argnums=(5,6,7))
def sghmc_kernel(key, params, param_grad, dt, alpha, L, val_and_grad_log_post, batch_size, X, y_data):
    N_data, dim = X.shape

    def body(carry, key):
        params, param_grad, v = carry
        subkey1, subkey2 = random.split(key)
        params = params + v
        v = v + dt*param_grad - alpha*v + jnp.sqrt(2*alpha*dt)*random.normal(subkey2, shape=(dim,))

        idx_batch = random.choice(key=subkey1, a=jnp.arange(N_data), shape=(batch_size,))
        _, param_grad = val_and_grad_log_post(params, X[idx_batch,:], y_data[idx_batch])
        return (params, param_grad, v), None

    subkey1, subkey2 = random.split(key)
    keys = random.split(subkey1, L)
    v = jnp.sqrt(dt)*random.normal(subkey2, shape=(dim,))
    (params, param_grad, _), _ = lax.scan(body, (params, param_grad, v), keys)
    return params, param_grad

@partial(jit, static_argnums=(1,4,5,6))
def sghmc_scan(key, num_samples, dt, alpha, L, val_and_grad_log_post, batch_size, X, y_data, x_0):

    @progress_bar_scan(num_samples)
    def body(carry, x):
        key = x[1]
        params, param_grad = carry
        params, param_grad = sghmc_kernel(key, params, param_grad, dt,
                                          alpha, L, val_and_grad_log_post, batch_size, X, y_data)
        return (params, param_grad), (params, param_grad)

    keys = random.split(key, num_samples)
    _, param_grad = val_and_grad_log_post(x_0, X, y_data)
    _, (samples, sample_grads) = lax.scan(body, (x_0, param_grad), (jnp.arange(num_samples), keys))
    return samples, sample_grads

def sghmc_time_budget(key, log_post, X, y_data, time_budget, dt, batch_size, x_0, alpha, L, verbose=0):
    N_data, dim = np.shape(X)
    samples = []
    grads = []
    X = jnp.array(X)
    y_data = jnp.array(y_data) # convert to jax array to avoid bug when doing `y_data[idx_batch]`
    params = x_0

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    _, param_grad = log_post(params, X[idx_batch,:], y_data[idx_batch])

    # run it once to compile the function
    wait_until_computed(sghmc_kernel(key, params, param_grad, dt,
              alpha, L, log_post, batch_size, X, y_data))

    start_time = time.time()
    if verbose==1:
        print(f"Running SGHMC for a budget of {time_budget} seconds with batch size {batch_size}")
    while time.time() - start_time < time_budget:
        key, subkey = random.split(key)
        params, param_grad = sghmc_kernel(key, params, param_grad, dt,
                              alpha, L, log_post, batch_size, X, y_data)
        samples.append(params)
        grads.append(param_grad)

    end_time = time.time()
    if verbose==1:
        print("Done.")
        print(f"Running time: {(end_time-start_time):.1f} sec")
        print(f"Generated {np.shape(samples)[0]} samples")

    return np.array(samples), np.array(grads)
