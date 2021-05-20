import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, partial, grad
from jax import nn, random
from jax import scipy
import tensorflow_datasets as tfds
import time
from .util import one_hot, read_idx
from config import BASE_DIR
"""
# Bayesian NN

Load data:
1. MNIST
2. not-MNIST
"""

# 1. MNIST
# ======
# load data
data_dir = '/tmp/tfds'

# Fetch full datasets for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
# mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True, as_supervised=True)
mnist_data, info = tfds.load(name="mnist", batch_size=-1, with_info=True, as_supervised=True)
mnist_data = tfds.as_numpy(mnist_data)
data_train, data_test = mnist_data['train'], mnist_data['test']


y_train = one_hot(data_train[1], 10)
y_test = one_hot(data_test[1], 10)

X_train = data_train[0]
X_train = X_train.reshape(X_train.shape[0], 28*28)

X_test = data_test[0]
X_test = X_test.reshape(X_test.shape[0], 28*28)

# Normalizing the RGB codes by dividing it to the max RGB value.
X_train = X_train/255
X_test = X_test/255
X_train = jnp.array(X_train)
y_train = jnp.array(y_train)
N_data = X_train.shape[0]


# ==========
# 2. notMNIST

y_train_notMNIST = one_hot(read_idx(f"{BASE_DIR}/data/notMNIST/train-labels-idx1-ubyte"), 10)
X_train_notMNIST = read_idx(f"{BASE_DIR}/data/notMNIST/train-images-idx3-ubyte")
X_train_notMNIST = X_train_notMNIST.reshape(X_train_notMNIST.shape[0], 28*28)

y_test_notMNIST = one_hot(read_idx(f"{BASE_DIR}/data/notMNIST/t10k-labels-idx1-ubyte"), 10)
X_test_notMNIST = read_idx(f"{BASE_DIR}/data/notMNIST/t10k-images-idx3-ubyte")
X_test_notMNIST = X_test_notMNIST.reshape(X_test_notMNIST.shape[0], 28*28)

# normalise the RGB codes
X_train_notMNIST = X_train_notMNIST/255
X_test_notMNIST = X_test_notMNIST/255
X_train_notMNIST = jnp.array(X_train_notMNIST)
y_train_notMNIST = jnp.array(y_train_notMNIST)
N_data_notMNIST = X_train_notMNIST.shape[0]
