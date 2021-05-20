
import jax.numpy as jnp
from jax import jit, vmap, partial, grad
from jax import random, lax
import numpy as np
import time
from util import progress_bar_scan


# from examples.bayesian_NN.NN_model import X_train, X_test, y_train, y_test
from examples.bayesian_NN.NN_model import init_network, predict, accuracy, log_post, grad_log_post, batch_loglik
from examples.bayesian_NN.util import flatten_NN_params


# key = random.PRNGKey(0)
# sizes = [784, 100, 10]
# params = init_network(key, sizes)


@partial(jit, static_argnums=(3,6,))
def kernel_NN(key, params, param_grad, grad_log_post, X, y, batch_size, dt):
    N, _ = jnp.shape(X)
    subkey1, *subkeys = random.split(key, len(params)*2 + 1) # 1 for the gradient, 2 for each element in params
    newparams = []
    for idx, ((w,b), (grad_w, grad_b)) in enumerate(zip(params, param_grad)):
        new_w = w + dt/2 * grad_w + jnp.sqrt(dt)*random.normal(subkeys[2*idx], shape=w.shape)
        new_b = b + dt/2 * grad_b + jnp.sqrt(dt)*random.normal(subkeys[2*idx+1], shape=b.shape)
        newparams.append((new_w, new_b))

    idx_batch = random.choice(key=subkey1, a=jnp.arange(N), shape=(batch_size,))
    new_param_grad = grad_log_post(newparams, X[idx_batch,:], y[idx_batch,:])
    return newparams, new_param_grad


@partial(jit, static_argnums=(1,3,5))
def scan_NN(key, Niter, dt, batch_size, params_IC, grad_log_post, X_train, y_train):

    @progress_bar_scan(Niter)
    def body_NN(carry, x):
        key, params, param_grad = carry
        key, subkey = random.split(key)
        params, param_grads = kernel_NN(subkey, params, param_grad,
                                        grad_log_post, X_train, y_train, batch_size, dt)
        return (key, params, param_grads), (params, param_grads)

    param_grads = grad_log_post(params_IC, X_train, y_train)
    carry = (key, params_IC, param_grads)
    carry, (samples, grads) = lax.scan(body_NN, carry, jnp.arange(Niter))
    return samples, grads



def sgld_NN(key, Niter, dt, batch_size, params_IC, grad_log_post, X_train, y_train, X_test=None, y_test=None,
                        save_rate=10, print_rate=100, return_percentage_accuracy=False):
    samples = []
    sample_grads = []
    logloss = []
    percentage_accuracy = []
    params = params_IC

    param_grad = grad_log_post(params, X_train[:batch_size], y_train[:batch_size])


    start = time.time()
    for i in range(Niter):
        key, subkey = random.split(key)
        params, param_grad = kernel_NN(subkey, params, param_grad, grad_log_post,X_train, y_train, batch_size, dt)
        if i%save_rate==0:
            if return_percentage_accuracy==True:
                logloss.append(-np.mean(batch_loglik(params, X_test, y_test)))
                percentage_accuracy.append(accuracy(params, X_test, y_test))
            else:
                pass
            samples.append(params)
            sample_grads.append(param_grad)
        if i%print_rate==0:
            print(f"\nIteration {i}/{Niter}")
            if return_percentage_accuracy==True:
                print(f"log-loss: {logloss[-1]:.2f}")
                print(f"percentage accuracy: {100*percentage_accuracy[-1]:.0f}%")
            else:
                pass

    # print("\nDone.")
    end = time.time()

    # print(f"Running time: {int((end-start)/60)} min, { (end-start)%60:.0f} sec")

    if return_percentage_accuracy==True:
        return samples, sample_grads, logloss, percentage_accuracy
    else:
        return samples, sample_grads


def sgld_NN_time_budget(key, time_budget, grad_log_post, dt, batch_size, params_IC, X_train, y_train, save_rate=10):
    """
    sgld for NN
    """
    samples = []
    sample_grads = []
    params = params_IC
    param_grads = grad_log_post(params, X_train, y_train)

    # compile the step function
    key, subkey = random.split(key)
    kernel_NN(key, params, param_grads, grad_log_post, X_train, y_train, batch_size, dt)[0][0][0][0][0].block_until_ready()

    start_time = time.time()
    i = 0
    while time.time()-start_time <time_budget:
        i+=1
        key, subkey = random.split(key)
        params, param_grads = kernel_NN(key, params, param_grads, grad_log_post, X_train, y_train, batch_size, dt)
        if i%save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
    return samples, sample_grads


@partial(jit, static_argnums=(3,6))
def kernel_NN_CV(key, params, param_grad, grad_log_post, X, y, batch_size, dt, mean_params, grad_at_mean):
    N, _ = jnp.shape(X)
    subkey1, *subkeys = random.split(key, len(params)*2 + 1) # 1 for the gradient, 2 for each element in params
    newparams = []
    for idx, ((w,b), (grad_w, grad_b)) in enumerate(zip(params, param_grad)):
        new_w = w + dt/2 * grad_w + jnp.sqrt(dt)*random.normal(subkeys[2*idx], shape=w.shape)
        new_b = b + dt/2 * grad_b + jnp.sqrt(dt)*random.normal(subkeys[2*idx+1], shape=b.shape)
        newparams.append((new_w, new_b))

    idx_batch = random.choice(key=subkey1, a=jnp.arange(N), shape=(batch_size,))
    minibatch_param_grad = grad_log_post(newparams, X[idx_batch,:], y[idx_batch,:])
    minibatch_mean_grad = grad_log_post(mean_params, X[idx_batch,:], y[idx_batch,:])
    new_param_grads = []

    for idx in range(len(minibatch_param_grad)):
        new_tuple = []
        for idx_tuple in [0,1]:
            new_tuple.append(grad_at_mean[idx][idx_tuple] + minibatch_param_grad[idx][idx_tuple] - minibatch_mean_grad[idx][idx_tuple])
        new_param_grads.append(tuple(new_tuple))
    return newparams, new_param_grads

def sgld_NN_time_budget_CV(key, time_budget, grad_log_post, dt, batch_size,
                           params_IC, X_train, y_train, mean_params, save_rate=10):
    """
    sgld for NN.
    grad_log_post should only return the gradient
    """
    logpostlist = []
    rmse_list = []
    samples = []
    sample_grads = []
    params = params_IC
    param_grads = grad_log_post(params, X_train, y_train)
    grad_at_MAP = grad_log_post(mean_params, X_train, y_train)
    # compile the step function
    key, subkey = random.split(key)
    _ = kernel_NN_CV(subkey, params, param_grads, grad_log_post, X_train,
                                           y_train, batch_size, dt, mean_params, grad_at_MAP)[0][0][0][0][0].block_until_ready()

    start_time = time.time()
    i = 0
    while time.time()-start_time <time_budget:
        i+=1
        key, subkey = random.split(key)
        params, param_grads = kernel_NN_CV(subkey, params, param_grads, grad_log_post, X_train,
                                           y_train, batch_size, dt, mean_params, grad_at_MAP)
        if i%save_rate==0:
            samples.append(params)
            sample_grads.append(param_grads)
    return samples, sample_grads
