# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)

import jax
import jax.numpy as jnp
from jax import random
import numpy as np # get rid of this eventually
from jax.experimental.ode import odeint
from functools import partial # reduces arguments to function by making some subset implicit

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from lnn import solve_dynamics
from utils import wrap_coords
#HACK
#from .physics import lagrangian_fn, analytical_fn
from physics import lagrangian_fn, analytical_fn


@partial(jax.jit, backend='cpu')
def get_trajectory(y0, times, use_lagrangian=False, **kwargs):
  # frames = int(fps*(t_span[1]-t_span[0]))
  # times = jnp.linspace(t_span[0], t_span[1], frames)
  # y0 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
  if use_lagrangian:
    y = solve_dynamics(lagrangian_fn, y0, t=times, is_lagrangian=True, rtol=1e-10, atol=1e-10, **kwargs)
  else:
    y = odeint(analytical_fn, y0, t=times, rtol=1e-10, atol=1e-10, **kwargs)
  return y

@partial(jax.jit, backend='cpu')
def get_trajectory_lagrangian(y0, times, **kwargs):
  return solve_dynamics(lagrangian_fn, y0, t=times, is_lagrangian=True, rtol=1e-10, atol=1e-10, **kwargs)

@partial(jax.jit, backend='cpu')
def get_trajectory_analytic(y0, times, **kwargs):
    return odeint(analytical_fn, y0, t=times, rtol=1e-10, atol=1e-10, **kwargs)

def get_dataset(seed=0, samples=1, t_span=[0,2000], fps=1, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)

    frames = int(fps*(t_span[1]-t_span[0]))
    times = np.linspace(t_span[0], t_span[1], frames)
    y0 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)

    xs, dxs = [], []
    vfnc = jax.jit(jax.vmap(analytical_fn))
    for s in range(samples):
      x = get_trajectory(y0, times, **kwargs)
      dx = vfnc(x)
      xs.append(x) ; dxs.append(dx)
        
    data['x'] = jax.vmap(wrap_coords)(jnp.concatenate(xs))
    data['dx'] = jnp.concatenate(dxs)
    data['t'] = times

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx', 't']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data
