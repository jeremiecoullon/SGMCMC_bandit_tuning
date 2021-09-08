import numpy as np
from jax import partial, jit, vmap
import jax.numpy as jnp
from jax import lax, random, grad

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


# ==========
# FSSD

def imq_kernel(x, y):
    c, beta = 1., -0.5
    return (c + jnp.dot(x-y, x-y))**beta

k_dx = grad(imq_kernel, 0)

def xi(x, grad_x, v):
    return grad_x*imq_kernel(x,v) + k_dx(x,v)

batch_xi = vmap(xi, in_axes=(None, None, 0))


def tau(x, grad_x, V):
    d, = x.shape
    J, dv = V.shape
    assert dv == d
    return batch_xi(x, grad_x, V).reshape(J*d)/jnp.sqrt(d*J)

@jit
def delta_fn(x, y, g_x, g_y, V):
    return jnp.dot(tau(x, g_x, V), tau(y, g_y, V))


@jit
def FSSD(sgld_samples, sgld_grads, V):
    """
    FSSD with imq kernel
    """
    N = sgld_samples.shape[0]

    batch_delta_fun_rows = jit(vmap(delta_fn, in_axes=(None,0,None,0, None)))

    def body_ksd(le_sum, x):
        my_sample, my_grad = x
        le_sum += jnp.sum(batch_delta_fun_rows(my_sample, sgld_samples, my_grad, sgld_grads, V))
        return le_sum, None

    le_sum, _ = lax.scan(body_ksd, 0., (sgld_samples, sgld_grads))
    return jnp.sqrt(le_sum)/N


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

# =====
# Slower version, for both high dimension and number of samples
# def build_h_fn(kernel):
#     k_dx = grad(kernel, 0)
#     k_dy = grad(kernel, 1)
#
#     k_dx_i = lambda x,y, i: grad(kernel, 0)(x,y)[i]
#     k_dxdy_i = lambda x,y, i: grad(k_dx_i, 1)(x,y, i)[i]
#     k_dxdy = lambda x,y: vmap(k_dxdy_i, in_axes=(None, None, 0))(x,y , jnp.arange(x.shape[0]))
#
#     @jit
#     def h_fn(x, y, dpx, dpy):
#         term1 = jnp.dot(dpx, dpy)*kernel(x,y)
#         term2 = jnp.dot(dpy, k_dx(x,y)) + jnp.dot(dpx, k_dy(x,y))
#         term3 = jnp.sum(k_dxdy(x,y))
#         return term1 + term2 + term3
#
#     return h_fn
#
#
# def imq_kernel(x,y):
#     c, beta = 1., -0.5
#     return (c + jnp.dot(x-y, x-y))**beta
#
# h_fn = build_h_fn(imq_kernel)
# batch_h_fn_rows = jit(vmap(h_fn, in_axes=(None,0,None,0)))
#
# @jit
# def imq_KSD(sgld_samples, sgld_grads):
#     """
#     KSD with imq kernel
#     """
#     N = sgld_samples.shape[0]
#
#     def body_ksd(le_sum, x):
#         my_sample, my_grad = x
#         le_sum += jnp.sum(batch_h_fn_rows(my_sample, sgld_samples, my_grad, sgld_grads))
#         return le_sum, None
#
#     le_sum, _ = lax.scan(body_ksd, 0., (sgld_samples, sgld_grads))
#     return jnp.sqrt(le_sum)/N
