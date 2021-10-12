import numpy as np
from jax import jit, vmap
import jax.numpy as jnp
from jax import lax, random, grad
from jax.experimental.optimizers import adam
from functools import partial

@jit
def k_0_fun(parm1, parm2, gradlogp1, gradlogp2, c=1., beta=-0.5):
    """
    KSD kernel with the 2 norm
    """
    diff = parm1-parm2
    dim = parm1.shape[0]
    base = (c**2 + jnp.dot(diff, diff))
    term1 = jnp.dot(gradlogp1,gradlogp2)*base**beta
    term2 = -2*beta * jnp.dot(gradlogp1, diff) * base**(beta-1)
    term3 = 2*beta * jnp.dot(gradlogp2, diff) * base**(beta-1)
    term4 = -2*dim*beta*(base**(beta-1))
    term5 = -4*beta* (beta-1)*base**(beta-2)*jnp.sum(jnp.square(diff))
    return term1 + term2 + term3 + term4 + term5


batch_k_0_fun_rows = jit(vmap(k_0_fun, in_axes=(None,0,None,0,None,None)))

@jit
def imq_KSD(sgld_samples, sgld_grads):
    """
    KSD with imq kernel
    """
    c, beta = 1., -0.5
    N = sgld_samples.shape[0]

    def body_ksd(le_sum, x):
        my_sample, my_grad = x
        le_sum += jnp.sum(batch_k_0_fun_rows(my_sample, sgld_samples, my_grad, sgld_grads, c, beta))
        return le_sum, None

    le_sum, _ = lax.scan(body_ksd, 0., (sgld_samples, sgld_grads))
    return jnp.sqrt(le_sum)/N



# ===================
# FSSD

def imq_kernel(x, y):
    c, beta = 1., -0.5
    return (c + jnp.dot(x-y, x-y))**beta

k_dx = grad(imq_kernel, 0)

def xi(x, grad_x, v):
    return grad_x*imq_kernel(x,v) + k_dx(x,v)

@jit
def FSSD(sgld_samples, sgld_grads, V):
    """
    FSSD with imq kernel. Linear implementation

    Start from the first bit of equation (2) in https://arxiv.org/pdf/1705.07673.pdf,
    and simplift the double sum so that you only need to calculate xi for each sample.
    So this is O(N) (with N the number of samples)
    """
    N = sgld_samples.shape[0]
    d = sgld_samples[0].shape[0]
    J = V.shape[0]

    def compute_xi_sum(v):
        batch_xi_sam = vmap(xi, in_axes=(0,0, None))
        all_xi = batch_xi_sam(sgld_samples, sgld_grads, v)
        return jnp.square(all_xi.sum(axis=0)) - jnp.sum(jnp.square(all_xi), axis=0)

    sum_over_J = jnp.sum(vmap(compute_xi_sum)(V))

    return sum_over_J/(d*J*N*(N-1))


@jit
def FSSD_O2(samples, grads, V):
    """
    FSSD with naive O(N^2) implementation
    Use the first bit of equation (2) in https://arxiv.org/pdf/1705.07673.pdf
    """
    N = samples.shape[0]
    d = samples[0].shape[0]
    J = V.shape[0]

    def kernel_xi(x, dx, y, dy, v):
        return jnp.dot(xi(x,dx,v), xi(y,dy,v))

    def sum_gram_mat(v):
        "Do the double sum over all samples (O(N^2)) minus the diagonal"
        le_sum = jnp.sum(vmap(lambda samples2, grads2:
                          vmap(lambda samples1, grads1:
                               kernel_xi(samples1, grads1, samples2, grads2, v))(samples, grads))(samples, grads))

        le_sum -= jnp.sum(vmap(kernel_xi, in_axes=(0,0,0,0,None))(samples, grads, samples, grads, v))
        return le_sum

    sum_over_J = jnp.sum(vmap(sum_gram_mat)(V))

    return sum_over_J/(N*(N-1)*d*J)


def get_test_locations(samples, J=10, key=random.PRNGKey(0)):
    _, dim = jnp.shape(samples)
    gauss_mean = jnp.mean(samples, axis=0)
    gauss_cov = jnp.cov(samples.T) + 1e-10*jnp.eye(dim)
    gauss_chol = jnp.linalg.cholesky(gauss_cov)
    # gauss_chol = jnp.diag(jnp.diag(jnp.linalg.cholesky(gauss_cov))) # diagonal version
    batch_get_samples = vmap(lambda k: jnp.dot(gauss_chol, random.normal(key, shape=(dim,))) + gauss_mean)

    # gauss_chol = jnp.std(samples, axis=0) # diagonal cholesky
    # batch_get_samples = vmap(lambda k: gauss_chol*random.normal(key, shape=(dim,)) + gauss_mean)
    V = batch_get_samples(random.split(key, J))
    return V


def opt_v(samples, grads, V, Niter):
    def opt_fssd_fn(v):
        return -FSSD(samples, grads, v)

    grad_opt_fn = jit(grad(opt_fssd_fn))
    init_fn, update, get_params = adam(1e-3)
    state = init_fn(V)

    def body(state, i):
        return update(i, grad_opt_fn(get_params(state)), state), None

    state, _ = lax.scan(body, state, jnp.arange(Niter))
    return get_params(state)



@partial(jit, static_argnums=(3,))
def FSSD_opt(sgld_samples, sgld_grads, V, Niter):
    V_opt = opt_v(sgld_samples, sgld_grads, V, Niter)
    return FSSD(sgld_samples, sgld_grads, V_opt)


# ==========
# linear KSD
@jit
def linear_imq_KSD(samples, grads):
    """
    linear KSD with imq kernel (Liu 2016)
    """
    c, beta = 1., -0.5
    N_2 = int(samples.shape[0]/2)
    batch_k_0_fun = vmap(k_0_fun, in_axes=(0,0,0,0,None, None))

    le_sum = jnp.sum(batch_k_0_fun(samples[::2][:N_2], samples[1::2][:N_2], grads[::2][:N_2], grads[1::2][:N_2], c, beta))
    return le_sum/N_2
