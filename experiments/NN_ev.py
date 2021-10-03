
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
from jax import grad, jit, random, partial, lax

import context
from models.bayesian_NN.NN_model import logprior, loglikelihood, init_network, get_accuracy_list_NN, get_ECE_MCE, ood_testing
from models.bayesian_NN.NN_data import X_train, y_train, X_test, y_test
from models.bayesian_NN.util import load_NN_MAP, add_noise_NN_params

from tuning.mamba import timed_sampler
from tuning.ksd import imq_KSD
from tuning.util import build_metric_curve_fn, get_mean_sd_ksd_curve
from util import wait_until_computed, flatten_param_list

from sgmcmcjax.kernels import build_sgld_kernel, build_sghmc_kernel, build_sgnht_kernel
from sgmcmcjax.kernels import build_sgldCV_kernel, build_sghmcCV_kernel, build_sgnhtCV_kernel

print("Evaluate NN samplers")

data = (X_train, y_train)

key = random.PRNGKey(0)

Niters = 1000
key, subkey = random.split(key)
params_IC = add_noise_NN_params(subkey, load_NN_MAP(), 1.)
batch_size = int(0.1*X_train.shape[0])

gridsearch_bs = int(0.1*X_train.shape[0])
centering_value = load_NN_MAP()

dt_baseline = 1/X_train.shape[0]



from sgmcmcjax.util import build_grad_log_post
from jax import vmap, jit

grad_log_post = build_grad_log_post(loglikelihood, logprior, data)



def get_fb_grads_NN(samples):
    """
    Return full-batch gradients for NN samples
    samples list of NN sample. Each NN sample is a list of tuples
    """
    return [grad_log_post(sam, X_train, y_train) for sam in samples]

def assess_NN_samples(samples, thin_step):
    print(f"Assessing the samples. Num samples: {len(samples)}")
    num_samples = len(samples)
    flat_samples = flatten_param_list(samples[::thin_step])
    ece_est, mce_est = get_ECE_MCE(flat_samples, X_test, y_test, M=10, pbar=False)
    accuracy_list = get_accuracy_list_NN(samples, thin_step, X_test, y_test)

    return ece_est, mce_est, num_samples, accuracy_list, samples[::thin_step]

def run_time_budget(key, time_budget, build_kernel, thin, dict_params, params_IC=load_NN_MAP()):
    timed_sampler_factory = timed_sampler(build_kernel)
    my_timed_sampler = timed_sampler_factory(**dict_params)
    samples, _ = my_timed_sampler(key, time_budget, params_IC)
    return assess_NN_samples(samples, thin)



print("build partial kernels")

partial_build_sgld = lambda dt, batch_size: build_sgld_kernel(dt, loglikelihood,
                                                              logprior, data, batch_size)
partial_build_sgldCV = lambda dt, batch_size: build_sgldCV_kernel(dt, loglikelihood,
                                                    logprior, data, batch_size, centering_value)

partial_build_sghmc = lambda dt, L, batch_size: build_sghmc_kernel(dt, L, loglikelihood,
                                                              logprior, data, batch_size)
partial_build_sghmcCV = lambda dt, L, batch_size: build_sghmcCV_kernel(dt, L, loglikelihood,
                                                              logprior, data, batch_size, centering_value)

partial_build_sgnht = lambda dt, batch_size: build_sgnht_kernel(dt, loglikelihood,
                                                              logprior, data, batch_size)
partial_build_sgnhtCV = lambda dt, batch_size: build_sgnhtCV_kernel(dt, loglikelihood,
                                                              logprior, data, batch_size, centering_value)


# time_budget = 3python

time_budget = 7
thin = 20


# 1. A long run
#
# - 1 million data points
# - 20sec budget
# - thin by 20

print("\n\nsgld")

sh_params = {'batch_size': 600, 'dt': 3.1622776e-06}
gridsearch_params = {'batch_size': gridsearch_bs, 'dt': 0.00031622776}
baseline_params = {'batch_size': gridsearch_bs, 'dt': dt_baseline}

print("ece_est, mce_est, num_samples:\n")
ece_est, mce_est, num_samples, accuracy_list_sgld_mamba, samples_sgld_mamba = run_time_budget(
                key, time_budget, partial_build_sgld, thin, sh_params)
print(f"MAMBA: {ece_est, mce_est, num_samples}")

ece_est, mce_est, num_samples, accuracy_list_sgld_gridsearch, samples_sgld_gridsearch = run_time_budget(
                key, time_budget, partial_build_sgld, thin, gridsearch_params)
print(f"gridsearch: {ece_est, mce_est, num_samples}")

ece_est, mce_est, num_samples, accuracy_list_sgld_baseline, samples_sgld_baseline = run_time_budget(
                key, time_budget, partial_build_sgld, thin, baseline_params)
print(f"baseline: {ece_est, mce_est, num_samples}")

burnin_sgld_mamba = 100
burnin_sgld_gridsearch = 0
burnin_sgld_baseline = 0

fig, ax = plt.subplots(3, figsize=(10,6))
plt.suptitle("accuracy", size=18)
ax[0].plot(accuracy_list_sgld_mamba)
ax[0].set_title("mamba")
ax[1].plot(accuracy_list_sgld_gridsearch)
ax[1].set_title("gridsearch")
ax[2].plot(accuracy_list_sgld_baseline)
ax[2].set_title("baseline")

plt.tight_layout()

print(np.mean(accuracy_list_sgld_mamba[burnin_sgld_mamba:]))
print(np.mean(accuracy_list_sgld_gridsearch[burnin_sgld_gridsearch:]))
print(np.mean(accuracy_list_sgld_baseline[burnin_sgld_baseline:]))

ood_entropy_samples_sgld_mamba = ood_testing(samples_sgld_mamba[burnin_sgld_mamba:], X_test)
ood_entropy_samples_sgld_gridsearch = ood_testing(samples_sgld_gridsearch[burnin_sgld_gridsearch:], X_test)
ood_entropy_samples_sgld_baseline = ood_testing(samples_sgld_baseline[burnin_sgld_baseline:], X_test)

# np.savetxt("outputs/NN_uncertainty_tests/ood/ood_sgld_mamba.txt", ood_entropy_samples_sgld_mamba)
# np.savetxt("outputs/NN_uncertainty_tests/ood/ood_sgld_gridsearch.txt", ood_entropy_samples_sgld_gridsearch)
# np.savetxt("outputs/NN_uncertainty_tests/ood/ood_sgld_baseline.txt", ood_entropy_samples_sgld_baseline)

# ood_entropy_samples_sgld_mamba = np.genfromtxt("outputs/NN_uncertainty_tests/ood/ood_sgld_mamba.txt")
# ood_entropy_samples_sgld_gridsearch = np.genfromtxt("outputs/NN_uncertainty_tests/ood/ood_sgld_gridsearch.txt")
# ood_entropy_samples_sgld_baseline = np.genfromtxt("outputs/NN_uncertainty_tests/ood/ood_sgld_baseline.txt")




plt.hist(ood_entropy_samples_sgld_mamba, label='mamba', alpha=0.5)
plt.hist(ood_entropy_samples_sgld_gridsearch, label='gridsearch', alpha=0.5)
plt.hist(ood_entropy_samples_sgld_baseline, label='baseline', alpha=0.5)

plt.legend()
plt.show()
