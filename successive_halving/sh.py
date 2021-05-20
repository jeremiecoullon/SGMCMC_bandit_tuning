import jax.numpy as jnp
from jax import random
from ksd import imq_KSD
import numpy as np
from collections import OrderedDict
import time


class Base_SGLD_SuccHalv_Time:
    """
    SLGD sampler base class for Successive Halving with a time budget. You can store samples and rerun the sampler
    """

    def __init__(self, key, logdt, x_0):
        self.logdt = logdt
        self.samples = None
        self.grads = None
        self.KSD = None
        self.key = key
        self.x_0 = x_0

    def _run_sampler(self, key, time_budget, dt):
        raise NotImplementedError("Need to add the specific sampler here")

    def get_last_sample(self):
        return self.samples[-1]

    def append_samples(self, new_samples, new_grads):
        if self.samples is not None:
            self.samples = jnp.concatenate([self.samples, new_samples])
        else:
            self.samples = new_samples
        if self.grads is not None:
            self.grads = jnp.concatenate([self.grads, new_grads])
        else:
            self.grads = new_grads

    def run(self, time_budget):
        # Start from last sample of the chain if it's already been run
        if self.samples is None:
            x_0 = self.x_0
        else:
            x_0 = self.get_last_sample()
        self.key, subkey = random.split(self.key)
        sgld_samples, sgld_grads = self._run_sampler(subkey, time_budget, x_0)
        self.append_samples(sgld_samples, sgld_grads)

    def get_ksd(self):
        self.KSD = imq_KSD(self.samples, self.grads)
        return self.KSD

    @property
    def len_samples(self):
        if self.samples is not None:
            return self.samples.shape[0]
        else:
            return 0



def create_T_list_2params(key, n, Ndata):
    """
    n: number of configurations in successive Halving
    Ndata: number of data points

    Creates T_list for 2 parameters: log_dt and b_s_r
    """
    subkey1, subkey2 = random.split(key, 2)

    logdt_array = random.uniform(subkey1, shape=(n,))*5-7
    smallest_b_s_r = -int(np.log10(Ndata)) + 1
    range_log_b_s_r = np.linspace(smallest_b_s_r, 0., 100)
    batch_size_ratio_array = 10**random.choice(subkey2, range_log_b_s_r , shape=(n,))
    T_list = [(logdt, b_s_r) for logdt, b_s_r in zip(logdt_array, batch_size_ratio_array)]
    return T_list

def create_T_list_sghmc(key, n, Ndata):
    """
    n: number of configurations in successive Halving
    Ndata: number of data points

    Creates T_list for 3 parameters: log_dt, b_s_r, and L
    """
    subkey1, subkey2, subkey3 = random.split(key, 3)
    logdt_array = random.uniform(subkey1, shape=(n,))*5-7

    smallest_b_s_r = -int(np.log10(Ndata)) + 1
    range_log_b_s_r = np.linspace(smallest_b_s_r, 0., 100)
    batch_size_ratio_array = 10**random.choice(subkey2, range_log_b_s_r , shape=(n,))
    L_leapfrog_array = random.choice(subkey3, np.arange(2, 30), shape=(n,))
    T_list = [pars for pars in zip(logdt_array, batch_size_ratio_array, L_leapfrog_array)]
    return T_list

def create_T_list_svrg(key, n, Ndata):
    """
    n: number of configurations in successive Halving
    Ndata: number of data points

    Creates T_list for 3 parameters: log_dt, b_s_r, and update_array
    """
    subkey1, subkey2, subkey3 = random.split(key, 3)
    logdt_array = random.uniform(subkey1, shape=(n,))*5-7

    smallest_b_s_r = -int(np.log10(Ndata)) + 1
    range_log_b_s_r = np.linspace(smallest_b_s_r, 0., 100)
    batch_size_ratio_array = 10**random.choice(subkey2, range_log_b_s_r , shape=(n,))
    update_rate_array = random.choice(subkey3, np.arange(10000), shape=(n,))

    T_list = [pars for pars in zip(logdt_array, batch_size_ratio_array, update_rate_array)]
    return T_list

def run_SH_time_budget_keep_3(key, r, n, eta, thin_step, SH_class, create_T_list, Ndata):
    """
    r: time budget
    """
    num_iters = int(np.log(n)/np.log(eta))
    total_budget = num_iters * n * r
    print(f"Number of iterations: {num_iters}")

    print(f"Total budget: around {int(total_budget)} sec\n=====")
    subkey1, subkey2 = random.split(key)
    T_list = create_T_list(subkey1, n, Ndata)

    sampler_dict = OrderedDict({pars: SH_class(subkey2, *pars, thin_step=thin_step) for pars in T_list})

    i = 0
    start = time.time()
    while True:
        n_i = int(n*eta**(-i))
        r_i = r*eta**i
        print(f"\nIteration {i}: {n_i } configurations with {r_i:.3f} seconds each")
        print(f"Combinations: {T_list}")
        for sampler in sampler_dict.values():
            sampler.run(r_i)
        # print(f"Number of samples at iteration {i}: {sampler_dict[T_list[0]].len_samples}")
        print("Number of samples: ")
        print([sampler_dict[par].len_samples for par in T_list])
        print("Calculating KSD for each sampler")
        print("Calculating KSD for each sampler")
        L_dict = OrderedDict({k: sampler_dict[k].get_ksd() for k in T_list})
        L_keys_sorted = sorted(L_dict, key=lambda k :L_dict[k]) # sorted keys (logdt, b_s_r, L) by increasing loss

        if len(T_list) > 5:
            idx_best = int(n_i/eta) if int(n_i/eta)>0 else 1
            T_list = L_keys_sorted[:idx_best]
            sampler_dict = OrderedDict({k:sampler_dict[k] for k in T_list})
            i += 1
        else:
            T_list = L_keys_sorted[:3]
            sampler_dict = OrderedDict({k:sampler_dict[k] for k in T_list})
            end = time.time()
            print(f"Running time: {end-start:.2f}sec")
            return sampler_dict
