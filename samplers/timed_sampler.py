from jax import grad, jit, random, partial, lax
from tqdm.auto import tqdm
import time

from util import wait_until_computed

def timed_sampler_factory(init_fn, my_kernel, get_params, save_rate=1):
    "Build generic timed sampler"
    kernel = jit(my_kernel)
    
    def sampler(key, time_budget, params):
        samples = []
        sgmcmc_grads = []
        key, subkey = random.split(key)
        state = init_fn(subkey, params)
        wait_until_computed(kernel(0, subkey, state))

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
