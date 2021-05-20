import time

import jax.numpy as jnp
from jax import partial, jit, vmap, grad, random, lax

from util import wait_until_computed
from examples.PMF.pmf_model import grad_log_post as grad_log_post_PMF


@partial(jit, static_argnums=(5,6,))
def sghmc_kernel_PMF(key, params, param_grad, dt, alpha, L, batch_size, R_train):
    N_data, _ = R_train.shape

    def body(carry, key):
        params, param_grad, v = carry
        subkey1, subkey2 = random.split(key)
        new_params = []
        new_v = []

        keys_noise = random.split(subkey2, len(params))
        for idx in range(len(params)):
            new_params.append(params[idx] + v[idx])
            new_v.append(v[idx] + dt*param_grad[idx] - alpha*v[idx]
                         + jnp.sqrt(2*alpha*dt)*random.normal(keys_noise[idx], shape=params[idx].shape))

        idx_batch = random.choice(key=subkey1, a=jnp.arange(N_data), shape=(batch_size,))
        new_param_grad = grad_log_post_PMF(new_params, R_train[idx_batch,:])
        # new_param_grad = grad_log_post_PMF(new_params, R_train)
        return (new_params, new_param_grad, new_v), None

    subkey1, subkey2 = random.split(key)
    keys = random.split(subkey1, L)

    keys_resample_momentum = random.split(subkey2, len(params))
    v = []
    for idx in range(len(params)):
        v.append(jnp.sqrt(dt)*random.normal(keys_resample_momentum[idx], shape=params[idx].shape))
    (params, param_grad, _), _ = lax.scan(body, (params, param_grad, v), keys)
    return params, param_grad


def run_sghmc_PMF_time_budget(key, time_budget, dt, batch_size, L, R_train, params, save_rate=50, print_rate=1000):
    alpha = 0.01
    samples = []
    sample_grads = []
    param_grads = grad_log_post_PMF(params, R_train[:batch_size])
    wait_until_computed(param_grads)

    iternum = 0
    # print("Compiling")
    _first_step = sghmc_kernel_PMF(key, params, param_grads, dt, alpha, L, batch_size, R_train)
    wait_until_computed(_first_step)
    # print("Running")

    start_time = time.time()
    while time.time() - start_time < time_budget:
        key, subkey = random.split(key)
        params, param_grads = sghmc_kernel_PMF(key, params, param_grads, dt, alpha, L, batch_size, R_train)
        iternum += 1
        if iternum % save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
        if iternum % print_rate ==0:
            print(f"Iteration {iternum}. Time: {time.time()-start_time:.0f} / {time_budget} sec")
    return samples, sample_grads


def run_sghmc_PMF(key, Nsamples, dt, batch_size, L, R_train, params, save_rate=50, print_rate=1000):
    alpha = 0.01
    samples = []
    sample_grads = []
    param_grads = grad_log_post_PMF(params, R_train[:batch_size])
    wait_until_computed(param_grads)

    iternum = 0

    _first_step = sghmc_kernel_PMF(key, params, param_grads, dt, alpha, L, batch_size, R_train)
    wait_until_computed(_first_step)

    for i in range(Nsamples):
        key, subkey = random.split(key)
        params, param_grads = sghmc_kernel_PMF(key, params, param_grads, dt, alpha, L, batch_size, R_train)
        iternum += 1
        if iternum % save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
        if iternum % print_rate ==0:
            print(f"Iteration {iternum}/{Nsamples}")
    return samples, sample_grads
