from jax.tree_util import tree_leaves, tree_flatten
from jax import jit, lax
import jax.numpy as jnp
from jax.experimental import host_callback
from tqdm.auto import tqdm

def wait_until_computed(x):
    for leaf in tree_leaves(x):
        leaf.block_until_ready()


@jit
def flatten_param_list(samples):
    """
    Turns a list of PyTrees into an array of shape (Nsamples, dim)
    If the samples are already jnp.ndarray, then simply return them
    """
    if isinstance(samples, jnp.ndarray):
        return samples
    else:
        flattened_samples = []
        for lesam in samples:
            flat_sample, _ = tree_flatten(lesam)
            flattened_samples.append(jnp.concatenate([elem.flatten() for elem in flat_sample]))
        return jnp.array(flattened_samples)
