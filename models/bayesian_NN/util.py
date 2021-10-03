import numpy as np
import jax.numpy as jnp
from jax import random
import struct
from config import BASE_DIR


def one_hot(x, k, dtype=jnp.float32):
    "Create a one-hot encoding of x of size k."
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def read_idx(filename):
    "to open idx file (for the notMNIST dataset)"
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def flatten_NN_params(params):
    """
    params: list of NN parameters
        Each param in a list of (mat, vect) for each layer
    """
    flattened_params = []
    for lesam in params:
        flattened_params.append(np.concatenate([np.concatenate([mat.flatten(), vect]) for mat, vect in lesam]))
    return jnp.array(flattened_params)
#
# def _flatten_jax(layer):
#     "Utility function for flatten_NN_params_jaxscan"
#     a10 = np.array([e.flatten() for e in layer[0]])
#     a11 = layer[1]
#     return np.concatenate([a10, a11], axis=1)
#
# def flatten_NN_params_jaxscan(params):
#     """
#     Flatten NN params that came out of `jax.lax.scan`
#     """
#     return jnp.concatenate([_flatten_jax(layer) for layer in params], axis=1)

def add_noise_NN_params(key, params, scale):
    keys = random.split(key, len(params))
    new_params = []
    for (W,b), subkey in zip(params, keys):
        k1,k2 = random.split(subkey)
        new_W = W + scale*random.normal(k1, shape=W.shape)
        new_b = b + scale*random.normal(k2, shape=b.shape)
        new_params.append((new_W, new_b))
    return new_params

# def load_NN_params():
#     params = np.load(f"{BASE_DIR}/data/NN_params.npy", allow_pickle=True)
#     return [tuple([jnp.array(e2) for e2 in e1]) for e1 in params]


def load_NN_MAP():
    params = np.load(f"{BASE_DIR}/parameters/NN_MAP.npy", allow_pickle=True)
    return [tuple([jnp.array(e2) for e2 in e1]) for e1 in params]
