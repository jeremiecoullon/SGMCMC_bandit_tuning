import numpy as np
import time

import jax.numpy as jnp
from jax import partial, jit, vmap, grad, random, lax
from util import wait_until_computed

from .NN_model import grad_log_post, init_zero_network



@partial(jit, static_argnums=(6,10))
def sgld_adam_step_NN(key, iter_num, params, param_grads, momentum, v_secondmom, grad_log_post,
                      dt, X, y, batch_size):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    N, _ = jnp.shape(X)
    subkey1, *subkeys = random.split(key, len(params)*2 + 1) # 1 for the gradient, 2 for each element in params

    newparams = []
    new_momentum = []
    new_v_secondmom = []
    m_hat = []
    v_hat = []
    adapt_dt = []

    for idx in range(len(param_grads)):
        mom_W = beta1*momentum[idx][0] + (1-beta1)*param_grads[idx][0]
        mom_b = beta1*momentum[idx][1] + (1-beta1)*param_grads[idx][1]
        new_momentum.append((mom_W, mom_b))

        vsec_W = beta2*v_secondmom[idx][0] + (1-beta2)*jnp.square(param_grads[idx][0])
        vsec_b = beta2*v_secondmom[idx][1] + (1-beta2)*jnp.square(param_grads[idx][1])
        new_v_secondmom.append((vsec_W, vsec_b))

        mhat_W = mom_W/(1-beta1**(iter_num+1))
        mhat_b = mom_b/(1-beta1**(iter_num+1))
        m_hat.append((mhat_W, mhat_b))

        vhat_W = vsec_W/(1-beta2**(iter_num+1))
        vhat_b = vsec_b/(1-beta2**(iter_num+1))
        v_hat.append((vhat_W, vhat_b))

        ad_dt_W = dt/(jnp.sqrt(vhat_W) + eps)
        ad_dt_b = dt/(jnp.sqrt(vhat_b) + eps)
        adapt_dt.append((ad_dt_W, ad_dt_b))

        key, subkey1, subkey2 = random.split(key, 3)
        new_W = params[idx][0] + (ad_dt_W/2)*mhat_W + jnp.sqrt(ad_dt_W)*random.normal(key=subkey1, shape=params[idx][0].shape)
        new_b = params[idx][1] + (ad_dt_b/2)*mhat_b + jnp.sqrt(ad_dt_b)*random.normal(key=subkey2, shape=params[idx][1].shape)
        newparams.append((new_W, new_b))


    idx_batch = random.choice(key=subkey1, a=jnp.arange(N), shape=(batch_size,))
    new_param_grad = grad_log_post(newparams, X[idx_batch,:], y[idx_batch,:])
    return newparams, new_param_grad, new_momentum, new_v_secondmom



def sgld_adam_NN_time_budget(key, time_budget, dt, batch_size, X_train, y_train, params, save_rate=10):
    samples = []
    sample_grads = []
    momentum = init_zero_network()
    v_secondmom = init_zero_network()
    param_grads = grad_log_post(params, X_train[:batch_size], y_train[:batch_size])
    wait_until_computed(param_grads)

    iternum = 0
    print("Compiling")
    _first_step = sgld_adam_step_NN(key, iternum, params, param_grads, momentum, v_secondmom, grad_log_post,
                   dt, X_train, y_train, batch_size)
    wait_until_computed(_first_step)
    print("Running")

    start_time = time.time()
    while time.time() - start_time < time_budget:
        key, subkey = random.split(key)
        params, param_grads, momentum, v_secondmom = sgld_adam_step_NN(key, iternum, params, param_grads,
                                momentum, v_secondmom, grad_log_post,
                               dt, X_train, y_train, batch_size)
        iternum += 1
        if iternum % save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
        if iternum % 2000==0:
            print(f"Iteration {iternum}. Time: {time.time()-start_time:.0f} / {time_budget} sec")
    return samples, sample_grads
