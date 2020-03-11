import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import numpy as np # get rid of this eventually
import argparse
from jax import jit
from jax.experimental.ode import odeint
from functools import partial # reduces arguments to function by making some subset implicit
from jax.experimental import stax
from jax.experimental import optimizers
import os, sys, time
sys.path.append('..')
sys.path.append('../experiment_dblpend/')
from lnn import lagrangian_eom_rk4, lagrangian_eom, unconstrained_eom
from data import get_dataset
from models import mlp as make_mlp
from utils import wrap_coords

from data import get_trajectory
from data import get_trajectory_analytic
from physics import analytical_fn

from jax.experimental.ode import odeint

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

# replace the lagrangian with a parameteric model
def learned_dynamics(params):
  @jit
  def dynamics(q, q_t):
#     assert q.shape == (2,)
    state = wrap_coords(jnp.concatenate([q, q_t]))
    return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
  return dynamics


from jax.experimental.stax import serial, Dense, Softplus, Tanh, elementwise, Relu


sigmoid = jit(lambda x: 1/(1+jnp.exp(-x)))
swish = jit(lambda x: x/(1+jnp.exp(-x)))
relu3 = jit(lambda x: jnp.clip(x, 0.0, float('inf'))**3)
Swish = elementwise(swish)
Relu3 = elementwise(relu3)

def extended_mlp(args):
    act = {
        'softplus': [Softplus, Softplus],
        'swish': [Swish, Swish],
        'tanh': [Tanh, Tanh],
        'tanh_relu': [Tanh, Relu],
        'soft_relu': [Softplus, Relu],
        'relu_relu': [Relu, Relu],
        'relu_relu3': [Relu, Relu3],
        'relu3_relu': [Relu3, Relu],
        'relu_tanh': [Relu, Tanh],
    }[args.act]
    hidden = args.hidden_dim
    output_dim = args.output_dim
    nlayers = args.layers
    
    layers = []
    layers.extend([
        Dense(hidden),
        act[0]
    ])
    for _ in range(nlayers - 1):
        layers.extend([
            Dense(hidden),
            act[1]
        ])
        
    layers.extend([Dense(output_dim)])
    
    return stax.serial(*layers)

vfnc = jax.jit(jax.vmap(analytical_fn))
vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic, mxsteps=100), (0, None), 0))
vget_unlimited = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic), (0, None), 0))

dataset_size=50
fps=10
samples=50



def new_get_dataset(rng, samples=1, t_span=[0, 10], fps=100, test_split=0.5, lookahead=1,
                    unlimited_steps=False, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs

    frames = int(fps*(t_span[1]-t_span[0]))
    times = jnp.linspace(t_span[0], t_span[1], frames)
    y0 = jnp.concatenate([
        jax.random.uniform(rng, (samples, 2))*2.0*np.pi,
        jax.random.uniform(rng+1, (samples, 2))*0.1
    ], axis=1)

    if not unlimited_steps:
        y = vget(y0, times)
    else:
        y = vget_unlimited(y0, times)
        
    #This messes it up!
#     y = np.concatenate(((y[..., :2]%(2*np.pi)) - np.pi, y[..., 2:]), axis=2)
    
    data['x'] = y[:, :-lookahead]
    data['dx'] = y[:, lookahead:] - data['x']
    data['x'] = jnp.concatenate(data['x'])
    data['dx'] = jnp.concatenate(data['dx'])
    data['t'] = jnp.tile(times[:-lookahead], (samples,))

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx', 't']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def make_loss(args):
    if args.loss == 'l1':
        @jax.jit
        def gln_loss(params, batch, l2reg):
            state, targets = batch#_rk4
            leaves, _ = tree_flatten(params)
            l2_norm = sum(jnp.vdot(param, param) for param in leaves)
            preds = jax.vmap(partial(lagrangian_eom_rk4, learned_dynamics(params), Dt=args.dt, n_updates=args.n_updates))(state)
            return jnp.sum(jnp.abs(preds - targets)) + l2reg*l2_norm/args.batch_size

    else:
        @jax.jit
        def gln_loss(params, batch, l2reg):
            state, targets = batch
            preds = jax.vmap(partial(lagrangian_eom_rk4, learned_dynamics(params)))(state)
            return jnp.sum(jnp.square(preds - targets)) + l2reg*l2_norm/args.batch_size
        
            
    return gln_loss

from copy import deepcopy as copy
from tqdm import tqdm

def train(args, model, data, rng):
    global opt_update, get_params, nn_forward_fn
    global best_params, best_loss
    best_params = None
    best_loss = np.inf
    best_small_loss = np.inf
    (nn_forward_fn, init_params) = model
    data = {k: jax.device_put(v) for k,v in data.items()}

    loss = make_loss(args)
    opt_init, opt_update, get_params = optimizers.adam(
    lambda t: jnp.select([t  < args.num_epochs//2,
                          t >= args.num_epochs//2],
                         [args.lr, args.lr2]))
    opt_state = opt_init(init_params)
    
    @jax.jit
    def update_derivative(i, opt_state, batch, l2reg):
        params = get_params(opt_state)
        return opt_update(i, jax.grad(loss, 0)(params, batch, l2reg), opt_state), params

    train_losses, test_losses = [], []
    
    for iteration in range(args.num_epochs):
        rand_idx = jax.random.randint(rng, (args.batch_size,), 0, len(data['x']))
        rng += 1
        
        batch = (data['x'][rand_idx], data['dx'][rand_idx])
        opt_state, params = update_derivative(iteration, opt_state, batch, args.l2reg)
        small_loss = loss(params, batch, 0.0)
        
        new_small_loss = False
        if small_loss < best_small_loss:
            best_small_loss = small_loss
            new_small_loss = True

        if new_small_loss or (iteration % 1000 == 0) or (iteration < 1000 and iteration % 100 == 0):
            params = get_params(opt_state)
            train_loss = loss(params, (data['x'], data['dx']), 0.0)/len(data['x'])
            train_losses.append(train_loss)
            test_loss = loss(params, (data['test_x'], data['test_dx']), 0.0)/len(data['test_x'])
            test_losses.append(test_loss)
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_params = params

            if jnp.isnan(test_loss).sum():
                break
            
            print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

    params = get_params(opt_state)
    return params, train_losses, test_losses, best_loss

from matplotlib import pyplot as plt
data = new_get_dataset(jax.random.PRNGKey(0), t_span=[0, dataset_size], fps=fps, samples=samples, test_split=0.9)

# args = ObjectView(dict(
    # num_epochs=100, #40000
    # loss='l1',
    # l2reg=1e-6,
    # act='softplus',
    # hidden_dim=500,
    # output_dim=1,
    # dt=1e-1,
    # layers=2,
    # lr=1e-3*0.5,
    # lr2=1e-4*0.5,
    # model='gln',
    # n_updates=3,
    # batch_size=32,
# ))

def test_args(args):
    print('Running on', args.__dict__)
    rng = jax.random.PRNGKey(0)
    init_random_params, nn_forward_fn = extended_mlp(args)
    _, init_params = init_random_params(rng+1, (-1, 4))
    model = (nn_forward_fn, init_params)

    result = train(args, model, data, rng+3)
    print(result[3], 'is the loss for', args.__dict__)

    if not jnp.isfinite(result[3]).sum():
        return {'status': 'fail', 'loss': float('inf')}
    return {'status': 'ok', 'loss': float(result[3])}

#test_args(args)
