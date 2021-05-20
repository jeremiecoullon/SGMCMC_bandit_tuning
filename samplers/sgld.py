# -*- coding: utf-8 -*-

import numpy as np
import jax.numpy as jnp
from jax import random, partial, jit, lax
import time
from util import progress_bar_scan, wait_until_computed


@partial(jit, static_argnums=(3,7,8,9))
def sgld_kernel(key, param, param_grad, log_post, dt, X, y_data, N_data, dim, batch_size):
    subkey1, subkey2 = random.split(key)
    param = param + (dt/2)*param_grad + jnp.sqrt(dt)*random.normal(key=subkey1, shape=(dim,))
    idx_batch = random.choice(key=subkey2, a=jnp.arange(N_data), shape=(batch_size,))
    _, param_grad = log_post(param, X[idx_batch,:], y_data[idx_batch])
    return param, param_grad

@partial(jit, static_argnums=(1,4,6))
def sgld_jaxloop(key, log_post, X, y_data, Nsamples, dt, batch_size, x_0):
    N_data, dim = X.shape

    @progress_bar_scan(Nsamples)
    def body(carry, x):
        key, param, param_grad = carry
        key, subkey = random.split(key)
        param, param_grad = sgld_kernel(subkey, param, param_grad,
                            log_post, dt, X, y_data, N_data, dim, batch_size)
        return (key, param, param_grad), (param, param_grad)

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    _, param_grad = log_post(x_0, X[idx_batch, :], y_data[idx_batch])
    carry = (key, x_0, param_grad)
    carry, (samples, grads) = lax.scan(body, carry, jnp.arange(Nsamples))
    return samples, grads

def sgld(key, log_post, X, y_data, Nsamples, dt, batch_size=None, x_0=None, print_rate=500):
    N_data, dim = np.shape(X)
    if batch_size is None:
        batch_size = int(N_data*0.1)
    else:
        pass
    samples = np.zeros((Nsamples, dim))
    grads = np.zeros((Nsamples, dim))
    X = jnp.array(X)
    y_data = jnp.array(y_data) # convert to jax array to avoid bug when doing `y_data[idx_batch]`
    if x_0 is not None:
        param = x_0
    else:
        param = jnp.zeros(dim)

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    _, param_grad = log_post(param, X[idx_batch,:], y_data[idx_batch])

    start = time.time()
    print(f"Running SGLD with batch size {batch_size}")
    for i in range(Nsamples):
        key, subkey = random.split(key)
        param, param_grad = sgld_kernel(subkey,
                                        param,
                                        param_grad,
                                        log_post,
                                        dt,
                                        X,
                                        y_data,
                                        N_data,
                                        dim,
                                        batch_size
                                        )
        samples[i] = param
        grads[i] = param_grad

        if i%print_rate==0:
            print(f"iteration {i}/{Nsamples}")

    end = time.time()
    print("Done.")
    print(f"Running time: {(end-start):.1f}sec")

    return samples, grads


def sgld_time_budget(key, log_post, X, y_data, time_budget, dt, batch_size, x_0, verbose=0):
    N_data, dim = np.shape(X)
    samples = []#np.zeros((Nsamples, dim))
    grads = []#np.zeros((Nsamples, dim))
    X = jnp.array(X)
    y_data = jnp.array(y_data) # convert to jax array to avoid bug when doing `y_data[idx_batch]`
    param = x_0

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    _, param_grad = log_post(param, X[idx_batch,:], y_data[idx_batch])

    # run it once to compile the function
    _ = sgld_kernel(subkey, param, param_grad, log_post,
                        dt, X, y_data, N_data, dim, batch_size)[0][0].block_until_ready()
                        
    start_time = time.time()
    if verbose==1:
        print(f"Running SGLD for a budget of {time_budget} seconds with batch size {batch_size}")
    while time.time() - start_time < time_budget:
        key, subkey = random.split(key)
        param, param_grad = sgld_kernel(subkey, param, param_grad, log_post,
                                        dt, X, y_data, N_data, dim, batch_size)
        samples.append(param)
        grads.append(param_grad)

    end_time = time.time()
    if verbose==1:
        print("Done.")
        print(f"Running time: {(end_time-start_time):.1f} sec")
        print(f"Generated {np.shape(samples)[0]} samples")

    return np.array(samples), np.array(grads)


def sgld_adam_time_budget(key, log_post, X, y_data, time_budget, dt, batch_size, x_0):
    N_data, dim = np.shape(X)
    samples = []
    grads = []
    momentum = jnp.zeros(x_0.shape)
    v_secondmom = jnp.zeros(x_0.shape)
    X = jnp.array(X)
    y_data = jnp.array(y_data) # convert to jax array to avoid bug when doing `y_data[idx_batch]`
    param = x_0

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    _, param_grad = log_post(param, X[idx_batch,:], y_data[idx_batch])

    # run it once to compile the function
    wait_until_computed(sgld_adam_step(key, 1, param, param_grad, momentum, v_secondmom,
                                           log_post, dt/2, X, y_data, batch_size))

    start_time = time.time()
    i = 0
    while time.time() - start_time < time_budget:
        key, subkey = random.split(key)
        param, param_grad, momentum, v_secondmom, adapt_dt = sgld_adam_step(key, i, param, param_grad,
                                                            momentum, v_secondmom,
                                                           log_post, dt/2, X, y_data, batch_size)
        samples.append(param)
        grads.append(param_grad)
        i += 1

    end_time = time.time()
    return np.array(samples), np.array(grads)




@partial(jit, static_argnums=(6,10))
def sgld_adam_step(key, iter_num, param, param_grad, momentum, v_secondmom, log_post, dt, X, y_data, batch_size):
    subkey1, subkey2 = random.split(key)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    momentum = beta1*momentum + (1-beta1)*param_grad
    v_secondmom = beta2*v_secondmom + (1-beta2)*jnp.square(param_grad)
    m_hat = momentum/(1-beta1**(iter_num+1))
    v_hat = v_secondmom/(1-beta2**(iter_num+1))
    adapt_dt = dt/(jnp.sqrt(v_hat) + eps)

    param = param + (adapt_dt/2)*m_hat + jnp.sqrt(adapt_dt)*random.normal(key=subkey1, shape=param.shape)


    idx_batch = random.choice(key=subkey2, a=jnp.arange(X.shape[0]), shape=(batch_size,))
    _, param_grad = log_post(param, X[idx_batch,:], y_data[idx_batch])
    return param, param_grad, momentum, v_secondmom, adapt_dt

@partial(jit, static_argnums=(1,4,6))
def run_adam_sgld(key, log_post, X, y_data, Nsamples, dt, batch_size, x_0):
    N_data, dim = X.shape

    @progress_bar_scan(Nsamples)
    def body(carry, iter_num):
        key, param, param_grad, momentum, v_secondmom = carry
        key, subkey = random.split(key)
        param, param_grad, momentum, v_secondmom, adapt_dt = sgld_adam_step(subkey, iter_num, param, param_grad,
                                                momentum, v_secondmom,
                                           log_post, dt, X, y_data, batch_size)
        return (key, param, param_grad, momentum, v_secondmom), (param, param_grad, adapt_dt)

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    _, param_grad = log_post(x_0, X[idx_batch, :], y_data[idx_batch])
    carry = (key, x_0, param_grad, jnp.zeros(x_0.shape), jnp.zeros(x_0.shape))
    carry, (samples, grads, adapt_dt_array) = lax.scan(body, carry, jnp.arange(Nsamples))
    return samples, grads, adapt_dt_array
