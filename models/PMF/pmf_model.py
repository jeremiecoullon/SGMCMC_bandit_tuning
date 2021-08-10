import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, random, partial

"""
Probabilistic Matrix Factorisation
"""


@jit
def loglikelihood(params, data):
    """
    log likelihood for a single data point
    alpha: precision of observational noise
    """
    alpha = 3.
    U, V, _, _, _, _ =  params
    i, j, r_ij = data
    return (-alpha/2)*(r_ij - jnp.dot(U[:,(i-1).astype(int)].T, V[:, (j-1).astype(int)]))**2

# batch_loglik = vmap(loglik, in_axes=(None, 0, None))

@jit
def lp_M(M, mu_M, lambda_M):
    "log prior for vectors U_i and V_j"
    return -0.5*jnp.linalg.multi_dot([M-mu_M, jnp.diag(jnp.exp(lambda_M)), M-mu_M]) + 0.5*jnp.sum(lambda_M)

batch_lp_M = jit(vmap(lp_M, in_axes=(1, None, None)))

def lp_lambda(log_lambda_i):
    """
    log_lambda_i: log of precision

    Prior is on the precision: lambda_i ~ Gamma(4, 5).
    So you can later run standard langevin update on it (as the support is R)
    """
    return 3*log_lambda_i - 5*jnp.exp(log_lambda_i) + log_lambda_i

batch_lp_lambda = vmap(lp_lambda, in_axes=(0))

def logprior(params):
    "Prior for U, V, mu_U, lambda_U, mu_V, lambda_V"
    U, V, mu_U, lambda_U, mu_V, lambda_V = params
    term_U = jnp.sum(batch_lp_M(U, mu_U, lambda_U))
    term_V = jnp.sum(batch_lp_M(V, mu_V, lambda_V))

    term_muU = -0.5*jnp.linalg.multi_dot([mu_U,jnp.diag(jnp.exp(lambda_U)), mu_U]) + 0.5*jnp.sum(lambda_U)
    term_muV = -0.5*jnp.linalg.multi_dot([mu_V,jnp.diag(jnp.exp(lambda_V)), mu_V]) + 0.5*jnp.sum(lambda_V)

    term_lambdaU = jnp.sum(batch_lp_lambda(lambda_U))
    term_lambdaV = jnp.sum(batch_lp_lambda(lambda_V))
    return term_U + term_V + term_muU + term_muV + term_lambdaU + term_lambdaV


# @jit
# def log_post(params, data):
#     Ndata = 80000 # dataset size
#     return logprior(params) + Ndata*jnp.mean(batch_loglik(params, data, 3.))
#
# grad_log_post = jit(grad(log_post))
# val_grad_logpost = jit(value_and_grad(log_post))
