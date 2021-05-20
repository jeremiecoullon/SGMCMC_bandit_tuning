# -*- coding: utf-8 -*-

import numpy as onp
import jax.numpy as np
from jax import random, partial, jit, lax
import time
from util import progress_bar_scan, wait_until_computed

@partial(jit, static_argnums=(3,5,6))
def mala_kernel(key, paramCurrent, paramGradCurrent, log_post, logpostCurrent, dt, dim):
    subkey1, subkey2 = random.split(key)
    paramProp = paramCurrent + dt*paramGradCurrent + np.sqrt(2*dt)*random.normal(key=subkey1, shape=(dim,))
    new_log_post, new_grad = log_post(paramProp)

    term1 = paramProp - paramCurrent - dt*paramGradCurrent
    term2 = paramCurrent - paramProp - dt*new_grad
    q_new = -0.25*(1/dt)*np.dot(term1, term1)
    q_current = -0.25*(1/dt)*np.dot(term2, term2)

    log_ratio = new_log_post - logpostCurrent + q_current - q_new
    acceptBool = np.log(random.uniform(key=subkey2)) < log_ratio
    paramCurrent = np.where(acceptBool, paramProp, paramCurrent)
    current_grad = np.where(acceptBool, new_grad, paramGradCurrent)
    current_log_post = np.where(acceptBool, new_log_post, logpostCurrent)
    accepts_add = np.where(acceptBool, 1,0)
    return paramCurrent, current_grad, current_log_post, accepts_add

@partial(jit, static_argnums=(1,2,3))
def mala_sampler_full_jax(key, val_and_grad_log_post, num_samples, dt, x_0):
    dim, = x_0.shape

    @progress_bar_scan(num_samples)
    def mala_step(carry, x):
        key, paramCurrent, gradCurrent, logpostCurrent, accepts = carry
        key, subkey = random.split(key)
        paramCurrent, gradCurrent, logpostCurrent, accepts_add = mala_kernel(subkey, paramCurrent, gradCurrent, val_and_grad_log_post, logpostCurrent, dt, dim)
        accepts += accepts_add
        return (key, paramCurrent, gradCurrent, logpostCurrent, accepts), (paramCurrent, gradCurrent)

    paramCurrent = x_0
    logpostCurrent, gradCurrent = val_and_grad_log_post(x_0)
    carry = (key, paramCurrent, gradCurrent, logpostCurrent, 0)
    (_, _, _, _, accepts), (samples, grads) = lax.scan(mala_step, carry, np.arange(num_samples))
    return samples, grads, 100*(accepts/num_samples)

def mala_sampler(key, log_post, N, dim, dt, x_0=None, print_rate=500):
    samples = onp.zeros((N, dim))
    grads = onp.zeros((N, dim))
    if x_0 is not None:
        paramCurrent = x_0
    else:
        paramCurrent = onp.zeros(dim)
    accepts = 0

    current_log_post, current_grad = log_post(paramCurrent)


    start = time.time()
    print(f"Running MALA for {N} iterations with dt={dt}")
    for i in range(N):
        key, subkey = random.split(key)
        paramCurrent, current_grad, current_log_post, accepts_add = mala_kernel(subkey,
                                                                        paramCurrent,
                                                                        current_grad,
                                                                        log_post,
                                                                        current_log_post,
                                                                        dt,
                                                                        dim)
        accepts += accepts_add
        samples[i] = paramCurrent
        grads[i] = current_grad
        if i%print_rate==0:
            print(f"Iteration {i}/{N}")

    end = time.time()
    print("Done.")
    print(f"Running time: {(end-start):.2f}sec")
    accept_rate = accepts/N * 100
    print(f"Acceptance rate: {accept_rate:.1f}%")

    return samples, grads



# Numpy version
# def mala_sampler(log_post, N, dim, dt, x_0=None, print_rate=500):
#     samples = np.zeros((N, dim))
#     grads = np.zeros((N, dim))
#     if x_0 is not None:
#         paramCurrent = x_0
#     else:
#         paramCurrent = np.zeros(dim)
#     accepts = 0
#
#     current_log_post, current_grad = log_post(paramCurrent)
#
#
#     start = time.time()
#     print(f"Running MALA for {N} iterations with dt={dt}")
#     for i in range(N):
#         paramProp = paramCurrent + dt*current_grad + np.sqrt(2*dt)*np.random.normal(size=dim)
#         new_log_post, new_grad = log_post(paramProp)
#
#         term1 = paramProp - paramCurrent - dt*current_grad
#         term2 = paramCurrent - paramProp - dt*new_grad
#         q_new = -0.25*(1/dt)*np.dot(term1, term1)
#         q_current = -0.25*(1/dt)*np.dot(term2, term2)
#
#         log_ratio = new_log_post - current_log_post + q_current - q_new
#         if np.log(np.random.uniform()) < log_ratio:
#             paramCurrent = paramProp
#             current_log_post = new_log_post
#             current_grad = new_grad
#             accepts += 1
#         samples[i] = paramCurrent
#         grads[i] = current_grad
#         if i%print_rate==0:
#             print(f"Iteration {i}/{N}")
#
#     end = time.time()
#     print("Done.")
#     print(f"Running time: {(end-start):.2f}sec")
#     accept_rate = accepts/N * 100
#     print(f"Acceptance rate: {accept_rate:.1f}%")
#
#     return samples, grads
