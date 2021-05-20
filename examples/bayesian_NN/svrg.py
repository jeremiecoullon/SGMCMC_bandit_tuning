import numpy as np
import time
from copy import deepcopy
import jax.numpy as jnp
from jax import partial, jit, vmap, grad, random, lax

from util import wait_until_computed



from .NN_data import X_train, X_test, y_train, y_test
from .NN_model import grad_log_post



@partial(jit, static_argnums=(5, 7, 11))
def step_NN_SVRG(key, params, param_grads, X_train, y_train, grad_log_post, dt, batch_size, theta_tilde, grad_tilde, idx_iter, update_rate):
    """
    Do a SVRG step for PMF (or any model who's parameters are list of arrays)
    """
    key, *subkeys = random.split(key, len(params)+1)
    newparams = []
    for idx in range(len(param_grads)):
        skey1, skey2 = random.split(subkeys[idx])
        new_W = params[idx][0] + (dt/2)*param_grads[idx][0] + jnp.sqrt(dt)*random.normal(skey1, shape=params[idx][0].shape)
        new_b = params[idx][1] + (dt/2)*param_grads[idx][1] + jnp.sqrt(dt)*random.normal(skey2, shape=params[idx][1].shape)
        newparams.append((new_W, new_b))

    idx_batch = random.choice(key, len(X_train), shape=(batch_size,))
    new_param_grads = []
    minibatch_param_grad = grad_log_post(newparams, X_train[idx_batch], y_train[idx_batch])
    minibatch_tilde_grad = grad_log_post(theta_tilde, X_train[idx_batch], y_train[idx_batch])
    for idx in range(len(param_grads)):
        new_W = grad_tilde[idx][0] + minibatch_param_grad[idx][0] - minibatch_tilde_grad[idx][0]
        new_b = grad_tilde[idx][1] + minibatch_param_grad[idx][1] - minibatch_tilde_grad[idx][1]
        new_param_grads.append((new_W, new_b))

    updateBool = idx_iter % update_rate == 0
    theta_tilde = lax.cond(updateBool,
                          lambda x: x,
                          lambda x: theta_tilde,
                          newparams)
    grad_tilde = lax.cond(updateBool,
                            lambda x: grad_log_post(x, X_train, y_train),
                            lambda x: grad_tilde,
                            newparams)
    return newparams, new_param_grads, theta_tilde, grad_tilde



def sgld_NN_SVRG(key, Niter, dt, batch_size, params_IC, update_rate, save_rate=10, print_rate=1000):
    """
    sgld for NN with control variates
    grad_log_post should only return the gradient
    """
    samples = []
    sample_grads = []
    params = deepcopy(params_IC)
    param_grads = grad_log_post(params, X_train[:batch_size], y_train[:batch_size])

    theta_tilde = deepcopy(params_IC)
    grad_tilde = grad_log_post(theta_tilde, X_train, y_train)

    first_iter = step_NN_SVRG(key, params, param_grads, X_train ,y_train, grad_log_post,
             dt, batch_size, theta_tilde, grad_tilde, 0, update_rate)
    wait_until_computed(first_iter)

    start = time.time()
    print(f"Running sgld for {Niter} iterations")
    for idx_iter in range(Niter):
        key, subkey = random.split(key)
        params, param_grads, theta_tilde, grad_tilde = step_NN_SVRG(key, params, param_grads, X_train ,y_train, grad_log_post,
                         dt, batch_size, theta_tilde, grad_tilde, idx_iter, update_rate)
        if idx_iter%save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
        if idx_iter%print_rate==0:
            print(f"Iteration {idx_iter}/{Niter}")

    end = time.time()
    print(f"\nRunning time {(end-start)/60:.2f}min")
    return samples

def sgld_NN_time_budget_svrg(key, time_budget, dt, batch_size, params_IC, update_rate, save_rate):
    """
    sgld for NN.
    grad_log_post should only return the gradient
    """
    samples = []
    sample_grads = []
    params = deepcopy(params_IC)
    param_grads = grad_log_post(params, X_train[:batch_size], y_train[:batch_size])

    # compile the step function
    key, subkey = random.split(key)
    theta_tilde = deepcopy(params_IC)
    grad_tilde = grad_log_post(theta_tilde, X_train, y_train)
    first_iter = step_NN_SVRG(key, params, param_grads, X_train ,y_train, grad_log_post,
             dt, batch_size, theta_tilde, grad_tilde, 0, update_rate)
    wait_until_computed(first_iter)

    start_time = time.time()
    i = 0
    while time.time()-start_time < time_budget:
        i+=1
        key, subkey = random.split(key)
        params, param_grads, theta_tilde, grad_tilde = step_NN_SVRG(key, params, param_grads, X_train ,y_train, grad_log_post,
                         dt, batch_size, theta_tilde, grad_tilde, i, update_rate)
        if i%save_rate==0:
            samples.append(params)
    return samples
