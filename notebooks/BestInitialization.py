import jax
import jax.numpy as jnp
import numpy as np # get rid of this eventually
import argparse
from jax import jit
from jax.experimental.ode import odeint
from functools import partial # reduces arguments to function by making some subset implicit

from jax.example_libraries import stax
from jax.experimental import optimizers

import os, sys, time
sys.path.append('..')


# ## Set up LNN:


sys.path.append('../experiment_dblpend/')

from lnn import raw_lagrangian_eom
from data import get_dataset
from models import mlp as make_mlp
from utils import wrap_coords


sys.path.append('../hyperopt')


from HyperparameterSearch import learned_dynamics


from HyperparameterSearch import extended_mlp


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


from data import get_trajectory


from data import get_trajectory_analytic


from physics import analytical_fn

vfnc = jax.jit(jax.vmap(analytical_fn))
vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic, mxstep=100), (0, None), 0))


import pickle as pkl


# ## Here are our model parameters


while True:
    hidden_dim = int(10**(np.random.rand()*1.5 + 1))
    layers = np.random.randint(2, 5)

    args = ObjectView({'dataset_size': 200,
     'fps': 10,
     'samples': 100,
     'num_epochs': 80000,
     'seed': 0,
     'loss': 'l1',
     'act': 'softplus',
     'hidden_dim': hidden_dim,
     'output_dim': 1,
     'layers': layers,
     'n_updates': 1,
     'lr': 0.001,
     'lr2': 2e-05,
     'dt': 0.1,
     'model': 'gln',
     'batch_size': 68,
     'l2reg': 5.7e-07,
    })
# args = loaded['args']
    rng = jax.random.PRNGKey(args.seed)


    from jax.experimental.ode import odeint


    from HyperparameterSearch import new_get_dataset


    from matplotlib import pyplot as plt


    vfnc = jax.jit(jax.vmap(analytical_fn, 0, 0))
    vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic, mxstep=100), (0, None), 0))

    batch = 60

    @jax.jit
    def get_derivative_dataset(rng):
        # randomly sample inputs

        y0 = jnp.concatenate([
            jax.random.uniform(rng, (batch, 2))*2.0*np.pi,
            (jax.random.uniform(rng+1, (batch, 2))-0.5)*10*2
        ], axis=1)
        
        return y0, vfnc(y0)


    best_params = None
    best_loss = np.inf


    init_random_params, nn_forward_fn = extended_mlp(args)
    import HyperparameterSearch
    HyperparameterSearch.nn_forward_fn = nn_forward_fn
    _, init_params = init_random_params(rng+1, (-1, 4))
    rng += 1
    model = (nn_forward_fn, init_params)
    opt_init, opt_update, get_params = optimizers.adam(args.lr)
    opt_state = opt_init(init_params)
    from jax.tree_util import tree_flatten
    from HyperparameterSearch import make_loss, train
    from copy import deepcopy as copy
# train(args, model, data, rng);
    from jax.tree_util import tree_flatten


# Current std:


    from jax.ops import index_update


    HyperparameterSearch.nn_forward_fn = nn_forward_fn


# ## Let's score the qdotdot output over normally distributed input for 256 batch size:


    from jax import grad, vmap


    normal = True
    n = 256

    @jax.jit
    def custom_init(stds, rng2):
        new_params = []
        i = 0
        for l1 in init_params:
            if (len(l1)) == 0: new_params.append(()); continue
            new_l1 = []
            for l2 in l1:
                if len(l2.shape) == 1:
                    new_l1.append(jnp.zeros_like(l2))
                else:
                    if normal:
                        new_l1.append(jax.random.normal(rng2, l2.shape)*stds[i])
#                     n1 = l2.shape[0]
#                     n2 = l2.shape[1]
#                     power = stds[0]
#                     base_scale = stds[1]
#                     s = base_scale/(n1+n2)**power
#                     new_l1.append(jax.random.normal(rng2, l2.shape)*s)
                    else:
                        new_l1.append(jax.random.uniform(rng2, l2.shape, minval=-0.5, maxval=0.5)*stds[i])
                    rng2+=1
                    i += 1

            new_params.append(new_l1)
            
        return new_params

    @jax.jit
    def j_score_init(stds, rng2):
        
        new_params = custom_init(stds, rng2)
        
        rand_input = jax.random.normal(rng2, [n, 4])
        rng2 += 1

        outputs = jax.vmap(
            partial(
            raw_lagrangian_eom,
            learned_dynamics(new_params)))(rand_input)[:, 2:]

        #KL-divergence to mu=0, std=1:
        mu = jnp.average(outputs, axis=0)
        std = jnp.std(outputs, axis=0)
        
        KL = jnp.sum((mu**2 + std**2 - 1)/2.0  - jnp.log(std))
        
        
        def total_output(p):
            return vmap(partial(raw_lagrangian_eom, learned_dynamics(p)))(rand_input).sum()

        d_params = grad(total_output)(new_params)
        
        i = 0
        for l1 in d_params:
            if (len(l1)) == 0: continue
            new_l1 = []
            for l2 in l1:
                if len(l2.shape) == 1: continue
                
                mu = jnp.average(l2)
                std = jnp.std(l2)
                KL += (mu**2 + std**2 - 1)/2.0 - jnp.log(std)
                
                #HACK
                desired_gaussian = jnp.sqrt(6)/jnp.sqrt(l2.shape[0] + l2.shape[1])
                scaled_std = stds[i]/desired_gaussian 
                #Avoid extremely large values
                KL += 0.1*(scaled_std**2/2.0 - jnp.log(scaled_std))
                i += 1

        return jnp.log10(KL)


    cur_std = jnp.array(
    [ 0.01]*(args.layers+1)
    )

    rng2 = jax.random.PRNGKey(0)


    j_score_init(cur_std, rng2)


# @jax.jit

    vv = jax.jit(vmap(j_score_init, (None, 0), 0))

    rng2 = jax.random.PRNGKey(0)
    def score_init(stds):
        global rng2
        stds = jnp.array(stds)
        stds = jnp.exp(stds)
        q75, q50, q25 = np.percentile(vv(stds, jax.random.split(rng2, num=10)), [75, 50, 25])
        rng2 += 30
        

        return q50, q75-q25


    score_init(cur_std)


# from bayes_opt import BayesianOptimization

# # Bounded region of parameter space
    pbounds = {'s%d'%(i,): (-15, 15) for i in range(len(cur_std))}


    def bb(**kwargs):
        out, std = score_init([kwargs[q] for q in ['s%d'%(i,) for i in range(len(cur_std))]])
#     if out is None or not out > -30:
#         return -30.0
        return -out, std


# Let's fit the best distribution:

# # Let's redo that with Bayes:

# # Bayesian:

# # Old stuff:


    import hyperopt
    from hyperopt import hp, fmin, tpe, Trials


    def run_trial(args):
        loss, std = bb(**args)
        if not np.isfinite(loss) or not np.isfinite(std):
            return {
            'status': 'fail', # or 'fail' if nan loss
            'loss': np.inf
        }

        return {
            'status': 'ok', # or 'fail' if nan loss
            'loss': -loss,
            'loss_variance': std,
        }


#TODO: Declare your hyperparameter priors here:
    space = {
        **{'s%d'%(i,): hp.normal('s%d'%(i,), -2, 5) for i in range(len(cur_std)-1)
        },
        **{'s%d'%(len(cur_std)-1,): hp.normal('s%d'%(len(cur_std)-1,), 3, 8)}
    }


    trials = Trials()


    best = fmin(run_trial,
        space=space,
        algo=tpe.suggest,
        max_evals=2500,
        trials=trials,
        verbose=1
        )


    def k(t):
        if 'loss' not in t['result']:
            return np.inf
        return t['result']['loss']

    sorted_trials = sorted(trials.trials, key=k)
    len(trials.trials)


    q = np.array(
        [[s['misc']['vals']['s%d'%(i,)][0] for i in range(len(cur_std))] for s in sorted_trials[:100]]
    )
    print(q[0], flush=True)


# ## 4 layers, 1000 hidden: {(4, 1000), (1000, 1000), (1000, 1000), (1000, 1)}
# 
# ## median top 10/2000: array([-1.47842217, -4.37217279, -3.37083752, 11.13480387])
# 
# (unconverged)
# 
# ## 4 layers, 100 hidden:  {(4, 100), (100, 100), (100, 100), (100, 1)}
# 
# ## median top 30/5000: array([-1.70680816, -2.40340615, -2.17201716, 10.55268474])
# 
# (unconverged)
# 
# ## 3 layers, 100 hidden:
# 
# ## median top 100/7000: array([-1.69875614, -2.74589338,  3.75818009])
# 
# (unverged converged)
# 
# ## 3 layers, 30 hidden:

# # Use Eureqa to get the scalings!


    simple_data = np.array(
    [
        [t['misc']['vals']['s%d'%(i,)][0] for i in range(len(cur_std))] + [t['result']['loss']]
    for t in trials.trials if 'loss' in t['result'] and np.isfinite(t['result']['loss'])])


# np.save('sdata.npy', simple_data)


    from sklearn.gaussian_process import GaussianProcessRegressor, kernels


    gp = GaussianProcessRegressor(alpha=3, n_restarts_optimizer=20, normalize_y=True)


    simple_data[:, -1].min()


    gp.fit(simple_data[:, :-1], simple_data[:, -1])


    print(args.layers+1, args.hidden_dim, q[gp.predict(q).argmin()], flush=True)

