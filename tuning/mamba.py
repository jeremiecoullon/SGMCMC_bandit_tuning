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


def run_MAMBA(key: PRNGKey, build_kernel: Callable, error_fn: Callable, T: float,
            params_IC: PyTree, grid_params: Union[Dict, None] = None, eta: int = 3) -> StateArm:
    """
    todo:
    - what if sampler gave nothing? 1. don't concat. 2. last sample.
    - number of iterations: check
    - keep best 3 rather than best 1 ?
    """
    list_hyperparams = create_grid(grid_params)
    timed_sampler_factory = timed_sampler(build_kernel)

    list_arms = []
    # initialisr arms
    for hyper_params in list_hyperparams:
        list_arms.append(StateArm(
            hyperparameters=hyper_params,
            run_timed_sampler=timed_sampler_factory(**hyper_params),
            last_sample=params_IC
        ))
    Niters = int(np.log(len(list_arms))/np.log(eta))

    start_time = time.time()
    for i in range(Niters):
        r = T/(len(list_arms)*Niters)
        for idx in tqdm(range(len(list_arms)), desc=f"Iteration {i+1}/{Niters}, time budget = {r:.1f} sec"):
            key, subkey = random.split(key)
            state = list_arms[idx]
            samples, grads = state.run_timed_sampler(subkey, r, state.last_sample)
            state = update_state_arm(state, samples, grads, error_fn)
            list_arms[idx] = state

        list_arms = sorted(list_arms, key=lambda s: s.metric)[:int(len(list_arms)/eta)]

    print(f"Running time: {time.time() - start_time:.1f} sec")
    assert len(list_arms) == 1
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
            state_hack = kernel(0, subkey, state)
            _ = kernel(1, subkey, state_hack)

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


def update_state_arm(state: StateArm, samples: List, grads: List, error_fn: Callable) -> StateArm:
    last_sample = samples[-1] if samples is not None else state.last_sample

    samples = flatten_param_list(samples)
    grads = flatten_param_list(grads)
    #concatenate
    if state.samples is not None:
        all_samples = jnp.concatenate([state.samples, samples])
    else:
        all_samples = samples
    if state.grads is not None:
        all_grads = jnp.concatenate([state.grads, grads])
    else:
        all_grads = grads
    metric = error_fn(samples, grads)
    state = StateArm(hyperparameters=state.hyperparameters,
                     run_timed_sampler=state.run_timed_sampler,
                     last_sample=last_sample,
                     samples=all_samples,
                     grads=all_grads,
                     metric=metric
                    )
    return state
