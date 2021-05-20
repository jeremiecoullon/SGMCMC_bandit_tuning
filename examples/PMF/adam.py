import numpy as np
import time

import jax.numpy as jnp
from jax import partial, jit, vmap, grad, random, lax
from util import wait_until_computed
from .util import init_PMF_zeros
from .pmf_model import grad_log_post as grad_log_post_PMF

@partial(jit, static_argnums=(6,9))
def sgld_adam_step_PMF(key, iter_num, params, param_grads, momentum, v_secondmom, log_post, dt, R_train, batch_size):
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    key, *subkeys = random.split(key, len(params)+1)
    newparams = []
    new_momentum = []
    new_v_secondmom = []
    m_hat = []
    v_hat = []
    adapt_dt = []

    for idx in range(len(param_grads)):
        new_momentum.append(beta1*momentum[idx] + (1-beta1)*param_grads[idx])
        new_v_secondmom.append(beta2*v_secondmom[idx] + (1-beta2)*jnp.square(param_grads[idx]))
        m_hat.append(new_momentum[idx]/(1-beta1**(iter_num+1)))
        v_hat.append(new_v_secondmom[idx]/(1-beta2**(iter_num+1)))
        adapt_dt.append(dt/(jnp.sqrt(v_hat[idx]) + eps))
        newparams.append(params[idx] + (adapt_dt[idx]/2)*m_hat[idx] +
                        jnp.sqrt(adapt_dt[idx])*random.normal(key=subkeys[idx], shape=params[idx].shape))

    idx_batch = random.choice(key=key, a=jnp.arange(R_train.shape[0]), shape=(batch_size,))
    new_param_grads = log_post(newparams, R_train[idx_batch])
    return newparams, new_param_grads, new_momentum, new_v_secondmom



def sgld_adam_PMF_time_budget(key, time_budget, dt, batch_size, R_train, params, save_rate=30):
    samples = []
    sample_grads = []
    momentum = init_PMF_zeros()
    v_secondmom = init_PMF_zeros()
    param_grads = grad_log_post_PMF(params, R_train[:batch_size])
    wait_until_computed(param_grads)

    iternum = 0
    print("Compiling")
    _first_step = sgld_adam_step_PMF(key, iternum, params, param_grads, momentum, v_secondmom, grad_log_post_PMF,
                   dt, R_train, batch_size)
    wait_until_computed(_first_step)
    print("Running")

    start_time = time.time()
    while time.time() - start_time < time_budget:
        key, subkey = random.split(key)
        params, param_grads, momentum, v_secondmom = sgld_adam_step_PMF(key, iternum, params, param_grads,
                                momentum,v_secondmom, grad_log_post_PMF,
                               dt, R_train, batch_size)
        iternum += 1
        if iternum % save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
        if iternum % 2000==0:
            print(f"Iteration {iternum}. Time: {time.time()-start_time:.0f} / {time_budget} sec")
    return samples, sample_grads
