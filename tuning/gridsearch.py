import time
import jax.numpy as jnp
from copy import deepcopy
import itertools
from sgmcmcjax.samplers import sgmcmc_sampler
from .sampler import timed_sampler

def create_grid(grid_params):
    """
    Args:
    -----
    grid_params: dict
        keys are the hyperparameter names. Values are the grid for that hyperparameter

    Returns:
    -------
    list_hyperparams: list
        List of grid points. Each element is a dictionary.
        key: hyperparameters. values: a single hyperparameter
    """
    if grid_params is None:
        grid_params = {'log_dt': -jnp.arange(2., 8.)}

    if "log_dt" in grid_params.keys():
        grid_params["dt"] = jnp.power(jnp.ones(len(grid_params['log_dt']))*10, grid_params.pop("log_dt"))

    list_hyperparams = [{k: v for k,v in zip(grid_params.keys(), elem)}
        for elem in itertools.product(*grid_params.values())]
    return list_hyperparams


def run_gridsearch(key, kernel, metric_fn, Niters, x_0, grid_params=None):
    list_hyperparams = create_grid(grid_params)
    build_sampler = sgmcmc_sampler(kernel)
    list_metric = []

    str_params = ', '.join(list(list_hyperparams[0].keys()))
    print(f"Running gridsearch for {str_params}\nNumber of grid points: {len(list_hyperparams)}")
    start_time = time.time()
    for hyper_params in list_hyperparams:
        print(hyper_params)
        my_sampler = build_sampler(**hyper_params, compiled=False)
        samples = my_sampler(key, Niters, x_0)
        list_metric.append(metric_fn(samples))
    print(f"Running time for gridsearch: {time.time() - start_time:.0f} sec")
    return list_hyperparams, list_metric


def run_gridsearch_time_budget(key, kernel, metric_fn, time_budget, x_0, grid_params=None):
    list_hyperparams = create_grid(grid_params)
    build_sampler = timed_sampler(kernel)
    list_metric = []

    str_params = ', '.join(list(list_hyperparams[0].keys()))
    print(f"Running gridsearch for {str_params}\nNumber of grid points: {len(list_hyperparams)}")
    start_time = time.time()
    for hyper_params in list_hyperparams:
        print(hyper_params)
        my_sampler = build_sampler(**hyper_params)
        samples, _ = my_sampler(key, time_budget, x_0)
        list_metric.append(metric_fn(samples))
    print(f"Running time for gridsearch: {time.time() - start_time:.0f} sec")
    return list_hyperparams, list_metric
