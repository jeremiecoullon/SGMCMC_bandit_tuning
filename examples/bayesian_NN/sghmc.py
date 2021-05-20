import numpy as np
import time

import jax.numpy as jnp
from jax import partial, jit, vmap, random, lax
from util import wait_until_computed

from examples.bayesian_NN.NN_data import X_train, X_test, y_train, y_test
from examples.bayesian_NN.util import load_NN_MAP
from examples.bayesian_NN.NN_model import grad_log_post
from copy import deepcopy


@partial(jit, static_argnums=(4,5,))
def sghmc_kernel_NN(key, params, param_grad, dt, L, batch_size, X_train, y_train):
    alpha = 0.01
    N_data = X_train.shape[0]

    def body(carry, key):
        params, param_grad, v = carry
        subkey1, subkey2 = random.split(key)
        new_params = []
        new_v = []

        keys_noise = random.split(subkey1, len(params))
        for idx in range(len(params)):
            new_p_W = params[idx][0] + v[idx][0]
            new_p_b = params[idx][1] + v[idx][1]
            new_params.append((new_p_W, new_p_b))

            k1, k2 = random.split(keys_noise[idx])
            new_v_W = v[idx][0] + dt*param_grad[idx][0] - alpha*v[idx][0] + jnp.sqrt(2*alpha*dt)*random.normal(k1, shape=params[idx][0].shape)
            new_v_b = v[idx][1] + dt*param_grad[idx][1] - alpha*v[idx][1] + jnp.sqrt(2*alpha*dt)*random.normal(k2, shape=params[idx][1].shape)
            new_v.append((new_v_W, new_v_b))

        idx_batch = random.choice(key=subkey2, a=jnp.arange(N_data), shape=(batch_size,))
        new_param_grad = grad_log_post(new_params, X_train[idx_batch,:], y_train[idx_batch])
        return (new_params, new_param_grad, new_v), None

    subkey1, subkey2 = random.split(key)
    keys = random.split(subkey1, L)

    keys_resample_momentum = random.split(subkey2, len(params))
    v = []
    for idx in range(len(params)):
        k1, k2 = random.split(keys_resample_momentum[idx])
        new_v_W = jnp.sqrt(dt)*random.normal(k1, shape=params[idx][0].shape)
        new_v_b = jnp.sqrt(dt)*random.normal(k2, shape=params[idx][1].shape)
        v.append((new_v_W, new_v_b))
    (params, param_grad, _), _ = lax.scan(body, (params, param_grad, v), keys)
    return params, param_grad



def sghmc_NN_time_budget(key, time_budget, dt, batch_size, x_0, L, save_rate=10):
    N_data, dim = np.shape(X_train)
    samples = []
    params = deepcopy(x_0)

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    param_grad = grad_log_post(params, X_train[idx_batch,:], y_train[idx_batch])

    # compile the kernel function
    wait_until_computed(sghmc_kernel_NN(key, params, param_grad, dt,
                                        L, batch_size, X_train, y_train))

    start_time = time.time()
    idx = 0
    while time.time() - start_time < time_budget:
        key, subkey = random.split(key)
        params, param_grad = sghmc_kernel_NN(key, params, param_grad, dt,
                                        L, batch_size, X_train, y_train)
        if idx % save_rate == 0:
            samples.append(params)
        idx += 1

    end_time = time.time()
    return samples



def run_sghmc_NN(key, Nsamples, dt, batch_size, x_0, L, save_rate=10):
    N_data, dim = np.shape(X_train)
    samples = []
    params = deepcopy(x_0)

    key, subkey = random.split(key)
    idx_batch = random.choice(key=subkey, a=jnp.arange(N_data), shape=(batch_size,))
    param_grad = grad_log_post(params, X_train[idx_batch,:], y_train[idx_batch])

    # compile the kernel function
    wait_until_computed(sghmc_kernel_NN(key, params, param_grad, dt,
                                        L, batch_size, X_train, y_train))

    start_time = time.time()
    for idx in range(Nsamples):
        key, subkey = random.split(key)
        params, param_grad = sghmc_kernel_NN(key, params, param_grad, dt,
                                        L, batch_size, X_train, y_train)
        if idx % save_rate == 0:
            samples.append(params)

    return samples
