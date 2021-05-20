import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, random, partial, lax
import time
from copy import deepcopy
from .util import init_params, rmse_PMF
from util import wait_until_computed, progress_bar_scan

@partial(jit, static_argnums=(4,6))
def step_sgld_PMF(key, params, param_grads, R_train, grad_log_post, dt, batch_size):
    """
    Do a sgld step for PMF (or any model who's parameters are list of arrays)
    """
    key, *subkeys = random.split(key, len(params)+1)
    newparams = []
    for idx in range(len(param_grads)):
        newparams.append(params[idx] + (dt/2)*param_grads[idx] + jnp.sqrt(dt)*random.normal(subkeys[idx], shape=params[idx].shape))

    idx_batch = random.choice(key, len(R_train), shape=(batch_size,))
    new_param_grads = grad_log_post(newparams, R_train[idx_batch])
    return newparams, new_param_grads

@partial(jit, static_argnums=(1,2,4))
def sgld_PMF_scan(key, Niter, grad_log_post, dt, batch_size, params_IC, R_train):

    @progress_bar_scan(Niter)
    def body(carry, x):
        key, params, param_grads = carry
        key, subkey = random.split(key)
        params, param_grads = step_sgld_PMF(subkey, params, param_grads, R_train, grad_log_post, dt, batch_size)
        return (key, params, param_grads),  (params, param_grads)
    carry = (key, params_IC, grad_log_post(params_IC, R_train))
    carry, (samples,grads) = lax.scan(body, carry, jnp.arange(Niter))
    return samples, grads


@partial(jit, static_argnums=(4,6))
def step_sgld_PMF_CV(key, params, param_grads, R_train, grad_log_post, dt, batch_size, mean_params, grad_at_mean):
    """
    Do a sgld step for PMF (or any model who's parameters are list of arrays)
    """
    key, *subkeys = random.split(key, len(params)+1)
    newparams = []
    for idx in range(len(param_grads)):
        newparams.append(params[idx] + (dt/2)*param_grads[idx] + jnp.sqrt(dt)*random.normal(subkeys[idx], shape=params[idx].shape))

    idx_batch = random.choice(key, len(R_train), shape=(batch_size,))
    new_param_grads = []
    minibatch_param_grad = grad_log_post(newparams, R_train[idx_batch])
    minibatch_mean_grad = grad_log_post(mean_params, R_train[idx_batch])
    for idx in range(len(param_grads)):
        new_param_grads.append(grad_at_mean[idx] + minibatch_param_grad[idx] - minibatch_mean_grad[idx])
    return newparams, new_param_grads

def sgld_PMF_time_budget_CV(key, time_budget, grad_log_post, dt,
                            batch_size, params_IC, R_train, mean_params, save_rate):
    """
    sgld for PMF.
    grad_log_post should only return the gradient
    """
    logpostlist = []
    rmse_list = []
    samples = []
    sample_grads = []
    if params_IC is None:
        params = init_params(key)
    else:
        params = params_IC
    param_grads = grad_log_post(params, R_train)
    grad_at_MAP = grad_log_post(mean_params, R_train)
    # compile the step function
    key, subkey = random.split(key)
    wait_until_computed(step_sgld_PMF_CV(subkey, params, param_grads, R_train, grad_log_post,
                         dt, batch_size, mean_params, grad_at_MAP))

    start_time = time.time()
    i = 0
    while time.time()-start_time < time_budget:
        i+=1
        key, subkey = random.split(key)
        params, param_grads = step_sgld_PMF_CV(subkey, params, param_grads, R_train, grad_log_post,
                                               dt, batch_size, mean_params, grad_at_MAP)
        if i%save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)

    return samples, sample_grads


def sgld_PMF_time_budget(key, time_budget, grad_log_post, dt, batch_size, params_IC, R_train, save_rate):
    """
    sgld for PMF.
    grad_log_post should only return the gradient
    """
    logpostlist = []
    rmse_list = []
    samples = []
    sample_grads = []
    if params_IC is None:
        params = init_params(key)
    else:
        params = params_IC
    param_grads = grad_log_post(params, R_train)

    # compile the step function
    key, subkey = random.split(key)
    _ = step_sgld_PMF(subkey, params, param_grads, R_train, grad_log_post, dt, batch_size)

    start_time = time.time()
    i = 0
    while time.time()-start_time < time_budget:
        i+=1
        key, subkey = random.split(key)
        params, param_grads = step_sgld_PMF(subkey, params, param_grads, R_train, grad_log_post, dt, batch_size)
        if i%save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
#     print(f"Done. Did  {i} iterations")
    return samples, sample_grads




def sgld_PMF(key, Niter, grad_log_post, dt, batch_size, params_IC, R_train, save_rate, print_rate):
    """
    sgld for PMF.
    grad_log_post should only return the gradient
    """
    logpostlist = []
    rmse_list = []
    samples = []
    sample_grads = []
    params = params_IC
    param_grads = grad_log_post(params, R_train)

    start = time.time()
    print(f"Running sgld for {Niter} iterations")
    for i in range(Niter):
        key, subkey = random.split(key)
        params, param_grads = step_sgld_PMF(subkey, params, param_grads, R_train, grad_log_post, dt, batch_size)
        if i%save_rate==0:
            # logpostlist.append(log_post(params, R_train))
            # rmse_list.append(rmse_PMF(params, R_test, mean_rating))
            samples.append(params)
            sample_grads.append(param_grads)
        if i%print_rate==0:
            print(f"Iteration {i}/{Niter}")

    end = time.time()
    print(f"\nRunning time {(end-start)/60:.2f}min")
    return samples, sample_grads


@partial(jit, static_argnums=(4, 6, 10))
def step_sgld_PMF_SVRG(key, params, param_grads, R_train, grad_log_post, dt, batch_size, theta_tilde, grad_tilde, idx_iter, update_rate):
    """
    Do a SVRG step for PMF (or any model who's parameters are list of arrays)
    """
    key, *subkeys = random.split(key, len(params)+1)
    newparams = []
    for idx in range(len(param_grads)):
        newparams.append(params[idx] + (dt/2)*param_grads[idx] + jnp.sqrt(dt)*random.normal(subkeys[idx], shape=params[idx].shape))

    idx_batch = random.choice(key, len(R_train), shape=(batch_size,))
    new_param_grads = []
    minibatch_param_grad = grad_log_post(newparams, R_train[idx_batch])
    minibatch_tilde_grad = grad_log_post(theta_tilde, R_train[idx_batch])
    for idx in range(len(param_grads)):
        new_param_grads.append(grad_tilde[idx] + minibatch_param_grad[idx] - minibatch_tilde_grad[idx])

    updateBool = idx_iter % update_rate == 0
    theta_tilde = lax.cond(updateBool,
                          lambda x: x,
                          lambda x: theta_tilde,
                          newparams)
    grad_tilde = lax.cond(updateBool,
                            lambda x: grad_log_post(x, R_train),
                            lambda x: grad_tilde,
                            newparams)
    return newparams, new_param_grads, theta_tilde, grad_tilde


def sgld_PMF_SVRG(key, Niter, grad_log_post, dt, batch_size, params_IC, R_train, update_rate, save_rate=10, print_rate=10):
    """
    sgld for PMF with control variates
    grad_log_post should only return the gradient
    """

    logpostlist = []
    rmse_list = []
    samples = []
    sample_grads = []
    params = deepcopy(params_IC)
    param_grads = grad_log_post(params, R_train[:batch_size])

    theta_tilde = deepcopy(params_IC)
    grad_tilde = grad_log_post(theta_tilde, R_train)
    start = time.time()
    print(f"Running sgld for {Niter} iterations")
    for idx_iter in range(Niter):
        key, subkey = random.split(key)
        params, param_grads, theta_tilde, grad_tilde = step_sgld_PMF_SVRG(subkey, params, param_grads, R_train,
                                 grad_log_post, dt, batch_size, theta_tilde,
                                 grad_tilde, idx_iter, update_rate)
        if idx_iter%save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
        if idx_iter%print_rate==0:
            print(f"Iteration {idx_iter}/{Niter}")

    end = time.time()
    print(f"\nRunning time {(end-start)/60:.2f}min")
    return samples, sample_grads

def sgld_PMF_time_budget_svrg(key, time_budget, grad_log_post, dt, batch_size, params_IC, R_train, update_rate, save_rate):
    """
    sgld for PMF.
    grad_log_post should only return the gradient
    """
    logpostlist = []
    rmse_list = []
    samples = []
    sample_grads = []
    params = params_IC
    param_grads = grad_log_post(params, R_train)

    # compile the step function
    key, subkey = random.split(key)
    theta_tilde = params_IC
    grad_tilde = grad_log_post(theta_tilde, R_train)
    wait_until_computed(step_sgld_PMF_SVRG(subkey, params, param_grads, R_train,
                         grad_log_post, dt, batch_size, theta_tilde,
                         grad_tilde, 1, update_rate))

    start_time = time.time()
    i = 0
    while time.time()-start_time < time_budget:
        i+=1
        key, subkey = random.split(key)
        params, param_grads, theta_tilde, grad_tilde = step_sgld_PMF_SVRG(subkey, params, param_grads, R_train,
                         grad_log_post, dt, batch_size, theta_tilde,
                         grad_tilde, i, update_rate)
        if i%save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
    return samples, sample_grads
