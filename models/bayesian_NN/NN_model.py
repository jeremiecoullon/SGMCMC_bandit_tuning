from tqdm.auto import tqdm
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, partial, grad
from jax import nn, random
from jax import scipy
import time

# from .util import flatten_NN_params
from util import flatten_param_list



"""
# Bayesian NN

Use Chris's review paper: https://arxiv.org/pdf/1907.06986.pdf
"""

from .NN_data import X_train
N_data = X_train.shape[0]

# ==========
# Functions to initialise parameters
# initialise params: list of tuples (W, b) for each layer
def random_layer(key, m, n, scale=1e-2):
    key, subkey = random.split(key)
    return (scale*random.normal(key, (n,m))), scale*random.normal(subkey, (n,))


def init_network(key, sizes=[784, 100, 10]):
    keys = random.split(key, len(sizes))
    return [random_layer(k,m,n) for k,m,n in zip(keys, sizes[:-1], sizes[1:])]

def zero_layer(m, n):
    return (jnp.zeros((n,m))), jnp.zeros((n,))

def init_zero_network():
    sizes = [784, 100, 10]
    return [zero_layer(m,n) for m,n in zip(sizes[:-1], sizes[1:])]

# ===========
# predict and accuracy functions
@jit
def predict(params, x):
    # per-example predictions
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = nn.softmax(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return nn.log_softmax(logits)

# =================
# Log-posterior

@jit
def loglikelihood(params, X, y):
    return jnp.sum(y*predict(params, X))

# batch_loglik = vmap(loglik, in_axes=(None, 0, 0), out_axes=0)

def logprior(params):
    logP = 0.0
    for w, b in params:
        logP += jnp.sum(scipy.stats.norm.logpdf(w))
        logP += jnp.sum(scipy.stats.norm.logpdf(b))
    return logP


# Accuracy stuff:

# Single point accuracy:
batch_predict = vmap(predict, in_axes=(None, 0))

@jit
def accuracy(params, X, y):
    target_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(batch_predict(params, X), axis=1)
    return jnp.mean(predicted_class == target_class)


def get_accuracy_list_NN(samples, thin_step, X_test, y_test):
    """
    samples: list of NN samples
    return list of accuracy for each sample.
    """
    list_acc = []

    for sam in tqdm(samples[::thin_step]):
        list_acc.append(accuracy(sam, X_test, y_test))

    return list_acc

# BNN acccuracy:
# For each data point: get average prediction over all samples
# Then get average accuracy for all data points
@jit
def unflatten_NN_param(flat_sample):
    """
    flat_sample: ndarray of size 79510. a flatten BNN sample
    """
    e1 = flat_sample[:78500]
    e2 = flat_sample[78500:]
    tuple1 = e1[:-100].reshape(100, 784), e1[-100:]
    tuple2 = e2[:-10].reshape(10, 100), e2[-10:]
    return [tuple1, tuple2]

@jit
def predict_flat_sample(flat_sample, x):
    return predict(unflatten_NN_param(flat_sample), x)

batch_predict_flat_sample = vmap(predict_flat_sample, in_axes=(0, None))

@jit
def accuracy_1_data_BNN_flat(flat_samples, x, y):
    mean_pred = jnp.mean(batch_predict_flat_sample(flat_samples, x), axis=0)
    predicted_class = jnp.argmax(mean_pred)
    target_class = jnp.argmax(y)
    return jnp.array(predicted_class == target_class, int)


batch_accuracy_BNN = vmap(accuracy_1_data_BNN_flat, in_axes=(None, 0,0,))

@jit
def accuracy_BNN_flat(flat_samples, X, y):
    return jnp.mean(batch_accuracy_BNN(flat_samples, X, y))

def accuracy_BNN(samples, X, y):
    """
    Accuracy on test dataset.

    For each data point, get the average prediction over all samples
    Get average accuracy over all data points
    """
    flat_samples = flatten_param_list(samples)
    return accuracy_BNN_flat(flat_samples, X, y)

@jit
def get_prediction(flat_samples, x, y):
    mean_pred = jnp.mean(batch_predict_flat_sample(flat_samples, x), axis=0)
    predicted_class = jnp.argmax(mean_pred)
    pred_prob = jnp.exp(mean_pred)[predicted_class]
    target_class = jnp.argmax(y)
    succ = jnp.array(predicted_class == target_class, int)
    return predicted_class, pred_prob, succ


batch_get_prediction = jit(vmap(get_prediction, in_axes=(None, 0, 0)))


def get_ECE_MCE(flat_samples, X_test, y_test, M, pbar=True):
    """
    Input: flat samples and test dataset
    Returns ECE and MCE
    M: number of bins
    """
    pred_classes, pred_probs, succ_array = batch_get_prediction(flat_samples, X_test, y_test)

    _, bins = np.histogram(pred_probs, bins=M)

    list_ece_bins = []
    mce_list = []
    if pbar == False:
        _tqdm = lambda x: x
    else:
        _tqdm = tqdm
    for idx in _tqdm(range(1, M+1)):

        cond_array = (pred_probs<bins[idx]) & (pred_probs>=bins[idx-1])

        pred_probs_bin = pred_probs[cond_array]
        pred_classes_bin = pred_classes[cond_array]
        succ_array_bins = succ_array[cond_array]

        accuracy_bin = np.mean(succ_array_bins)
        average_confidence_bin = np.mean(pred_probs_bin)
        ece_bin = pred_probs_bin.shape[0] * abs(accuracy_bin - average_confidence_bin)
        mce_list.append(abs(accuracy_bin - average_confidence_bin))

        list_ece_bins.append(ece_bin)

    ece_NN = np.sum(list_ece_bins)/X_test.shape[0]
    return ece_NN, max(mce_list)


@jit
def _ood_testing(flat_samples, X):
    "OOD testing for flattened NN samples"

    @jit
    def entropy(logp_array):
        return jnp.sum(-logp_array*jnp.exp(logp_array))

    def entropy_1_data_pt(flat_samples, x):
        preds = jnp.mean(batch_predict_flat_sample(flat_samples, x), axis=0)
        return entropy(preds)

    batch_entropy = vmap(entropy_1_data_pt, in_axes=(None, 0))
    return batch_entropy(flat_samples, X)

def ood_testing(samples, X):
    "OOD testing. samples should be a list of tuples of arrays"
    flat_samples = flatten_param_list(samples)
    return _ood_testing(flat_samples, X)
