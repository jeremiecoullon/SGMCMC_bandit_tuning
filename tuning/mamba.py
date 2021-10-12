from typing import NamedTuple, Callable, Dict, Union, Any, List
from tqdm.auto import tqdm
from sgmcmcjax.types import PyTree, PRNGKey
import time
import functools

import numpy as np
from jax import random, jit
import jax.numpy as jnp

from .gridsearch import create_grid
from util import flatten_param_list, wait_until_computed

class StateArm(NamedTuple):
    hyperparameters: Dict
    run_timed_sampler: Callable
    last_sample: PyTree
    samples: Union[jnp.ndarray, None] = None
    grads: Union[jnp.ndarray, None] = None
    metric: Any = jnp.inf

    def __str__(self):
        return f"Hyperparams: {self.hyperparameters}., sample shape={self.samples.shape}. metric: {self.metric:.0f}"


def run_MAMBA(key: PRNGKey, build_kernel: Callable, error_fn: Callable, R: float,
            params_IC: PyTree, grid_params: Union[Dict, None] = None, get_fb_grads = None, eta: int = 3) -> StateArm:
    """
    R: running time (sec) of longest sampler

    todo:
    - what if sampler gave nothing in the given time budget? 1. don't concat. 2. last sample.
    - number of iterations: check manually
    """
    list_hyperparams = create_grid(grid_params)
    timed_sampler_factory = timed_sampler(build_kernel)

    list_arms = []
    # initialise arms
    for hyper_params in list_hyperparams:
        list_arms.append(StateArm(
            hyperparameters=hyper_params,
            run_timed_sampler=timed_sampler_factory(**hyper_params),
            last_sample=params_IC
        ))

    Niters = int(np.log(len(list_arms))/np.log(eta))
    r_0 = R/sum([eta**i for i in range(Niters)])
    T = r_0 * len(list_arms)*Niters # total time budget over all samplers

    start_time_mamba = time.time()
    for i in range(Niters):
        r = T/(len(list_arms)*Niters)
        for idx in tqdm(range(len(list_arms)), desc=f"Iteration {i+1}/{Niters}, {len(list_arms)} arms, time budget = {r:.2f} sec"):
            key, subkey = random.split(key)
            state = list_arms[idx]
            samples, grads = state.run_timed_sampler(subkey, r, state.last_sample)
            state = update_state_arm(state, samples, grads, error_fn, get_fb_grads)
            print(state)
            list_arms[idx] = state

        list_arms = sorted(list_arms, key=lambda arm: arm.metric)[:int(len(list_arms)/eta)]
        print(f"Number of samples: {[arm.samples.shape[0] for arm in list_arms]}")

    wait_until_computed(list_arms[0].metric)
    print(f"Running time: {(time.time() - start_time_mamba):.1f} sec")
    assert len(list_arms) < 3
    return list_arms[0]


def timed_sampler(build_kernel_fn):
    """
    Decorator that turns a kernel factory (from `sgmcmcjax`) into a timed sampler factory
    """
    @functools.wraps(build_kernel_fn)
    def wrapper(*args, **kwargs):
        save_rate = kwargs.pop('save_rate', 1)
        init_fn, my_kernel, get_params = build_kernel_fn(*args, **kwargs)
        kernel = jit(my_kernel)

        def sampler(key, time_budget, params):
            samples = []
            sgmcmc_grads = []
            key, subkey = random.split(key)
            state = init_fn(subkey, params)
            # wait_until_computed(kernel(0, subkey, state))
            # hack: need this to compile the first 2 iterations of sgnht
            # _ = kernel(1, subkey, kernel(0, subkey, state))
            wait_until_computed(kernel(1, subkey, kernel(0, subkey, state)))

            start_time = time.time()
            i = 0
            while time.time() - start_time < time_budget:
                key, subkey = random.split(key)
                state = kernel(i, subkey, state)
                if i%save_rate==0:
                    samples.append(get_params(state))
                    sgmcmc_grads.append(state.param_grad)
                i += 1

            return samples, sgmcmc_grads
        return sampler
    return wrapper


def update_state_arm(state: StateArm, samples: List, grads: List, error_fn: Callable, get_fb_grads=None) -> StateArm:
    last_sample = samples[-1] if samples is not None else state.last_sample

    if get_fb_grads:
        samples, grads = get_fb_grads(samples)

    samples = flatten_param_list(samples)
    grads = flatten_param_list(grads)
    # concatenate
    if state.samples is not None:
        all_samples = jnp.concatenate([state.samples, samples])
    else:
        all_samples = samples
    if state.grads is not None:
        all_grads = jnp.concatenate([state.grads, grads])
    else:
        all_grads = grads
    _metric = error_fn(samples, grads)
    metric = _metric if not jnp.isnan(_metric) else jnp.inf
    state = StateArm(hyperparameters=state.hyperparameters,
                     run_timed_sampler=state.run_timed_sampler,
                     last_sample=last_sample,
                     samples=all_samples,
                     grads=all_grads,
                     metric=metric
                    )
    return state
