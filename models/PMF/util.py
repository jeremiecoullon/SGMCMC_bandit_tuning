import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, random, partial
from tqdm.auto import tqdm

from config import BASE_DIR

"""
PMF util and data
data: the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/))
"""

R_train = np.genfromtxt(f"{BASE_DIR}/data/MovieLens/train.dat")
R_test = np.genfromtxt(f"{BASE_DIR}/data/MovieLens/test.dat")

mean_rating = jnp.mean(R_train[:,2])
R_train[:,2] = R_train[:,2] - mean_rating
R_train = jnp.array(R_train)

D = 20
N = 943
M = 1682

def init_params(key, D=D, N=N, M=M):
    subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = random.split(key, 6)

    U = random.normal(subkey1, shape=(D, N))
    V = random.normal(subkey2, shape=(D, M))
    mu_U = random.normal(subkey3, shape=(D,))
    lambda_U = random.normal(subkey4, shape=(D,))
    mu_V = random.normal(subkey5, shape=(D,))
    lambda_V = random.normal(subkey6, shape=(D,))
    params = [U, V, mu_U, lambda_U, mu_V, lambda_V]
    return params

def init_PMF_zeros(D=D, N=N, M=M):

    U = jnp.zeros(shape=(D, N))
    V = jnp.zeros(shape=(D, M))
    mu_U = jnp.zeros(shape=(D,))
    lambda_U = jnp.zeros(shape=(D,))
    mu_V = jnp.zeros(shape=(D,))
    lambda_V = jnp.zeros(shape=(D,))
    params = [U, V, mu_U, lambda_U, mu_V, lambda_V]
    return params

key = random.PRNGKey(0)
params = init_params(key)


@jit
def _predict_loss(U, V, data, mean_rating):
    "Get the predictive RMSE for a single data point and a single sample"
    i, j, r_ij = data
    pred = jnp.dot(U[:,(i-1).astype(int)].T, V[:, (j-1).astype(int)]) + mean_rating
    pred = jnp.where(pred>5, 5, pred)
    pred = jnp.where(pred<1, 1, pred)
    return (pred-r_ij)**2
    # return pred

def rmse_PMF_1data(U_samples, V_samples, data, mean_rating):
    """
    Get predictive RMSE for a single data point by averaging over the posterior samples for U and V

    U_samples: array of U matrices (users)
    V_samples: array of V matrices (films)
    data: a single data point (user_idx, film_idx, rating)
    mean_rating: float
    """
    # batch over samples (ie: matrices U and V)
    _vmap_predict_dot = vmap(_predict_loss, in_axes=(0, 0, None, None))
    return jnp.mean(_vmap_predict_dot(U_samples, V_samples, data, mean_rating))
    # mean_pred = jnp.mean(_vmap_predict_dot(U_samples, V_samples, data, mean_rating))
    # i, j, r_ij = data
    # return (mean_pred - r_ij)**2


@jit
def rmse_PMF(U_samples, V_samples, data, mean_rating):
    """
    Gets predictive RMSE for all data points


    U_samples: array of U matrices (users)
    V_samples: array of V matrices (films)
    data: Dataset. Dimension: (N, 3) with N number of data points. (user_idx, film_idx, rating)
    mean_rating: float
    """
    batch_rmse = vmap(rmse_PMF_1data, in_axes=(None, None, 0, None))
    return jnp.mean(batch_rmse(U_samples, V_samples, data, mean_rating))

def get_rmse_list(samples, R_test, mean_rating):
    U_samples = np.array([elem[0] for elem in samples])
    V_samples = np.array([elem[1] for elem in samples])

    rmse_list = []
    num_samples = len(samples)

    for idx in tqdm(range(num_samples)):
        Usam = U_samples[idx][np.newaxis]
        Vsam = V_samples[idx][np.newaxis]
        rmse_list.append(rmse_PMF(Usam, Vsam, R_test, mean_rating))
    return rmse_list

def load_PMF_params():
    return [np.genfromtxt(f"{BASE_DIR}/data/PMF_params/PMF_{idx}_IC.txt") for idx in range(6)]

def load_PMF_mean():
    return [np.genfromtxt(f"{BASE_DIR}/data/PMF_mean_params/PMF_{idx}_mean.txt") for idx in range(6)]

def load_PMF_MAP():
    return [np.genfromtxt(f"{BASE_DIR}/data/PMF_MAP_params/PMF_{idx}_MAP.txt") for idx in range(6)]

def save_PMF_params(params):
    for idx, par in enumerate(params):
        np.savetxt(f"{BASE_DIR}/data/PMF_mean_params/PMF_{idx}_mean.txt", par)

def load_PMF_NUTS_stds():
    return [np.genfromtxt(f"{BASE_DIR}/data/PMF_NUTS_stds/PMF_{idx}_std.txt") for idx in range(6)]

def flatten_PMF_params(samples):
    """
    samples: list of PMF samples
        Each sample is a list of arrays
    """
    flattened_samples = []
    for lesam in samples:
        flattened_samples.append(jnp.concatenate([elem.flatten() for elem in lesam]))
    return jnp.array(flattened_samples)


def _flatten_PMF(parm):
    "Utility function for flatten_PMF_jax_scan"
    return jnp.array([e.flatten() for e in parm])

def flatten_PMF_jax_scan(params):
    """
    Flatten PMF params that came out of `jax.lax.scan`
    """
    return jnp.concatenate([_flatten_PMF(parm) for parm in params], axis=1)
