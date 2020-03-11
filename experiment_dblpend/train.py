# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)

import jax
import jax.numpy as jnp
import numpy as np # get rid of this eventually
import argparse
from jax.experimental.ode import odeint
from functools import partial # reduces arguments to function by making some subset implicit

from jax.experimental import stax
from jax.experimental import optimizers

import os, sys, time
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from lnn import lagrangian_eom, unconstrained_eom
from .data import get_dataset
from models import mlp
from utils import wrap_coords

def get_args():
    return {'input_dim': 4,
           'hidden_dim': 128,
           'output_dim': 4,
           'dataset_size': 3000,
           'learn_rate': 1e-3,
           'batch_size': 100,
           'test_every': 10,
           'num_batches': 500,
           'name': 'dblpend',
           'model': 'baseline_nn',
           'verbose': True,
           'seed': 1,
           'save_dir': '.'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


# replace the lagrangian with a parameteric model
def learned_dynamics(params):
  def dynamics(q, q_t):
    assert q.shape == (2,)
    state = wrap_coords(jnp.concatenate([q, q_t]))
    return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
  return dynamics

@jax.jit
def gln_loss(params, batch, time_step=None):
  state, targets = batch
  preds = jax.vmap(partial(lagrangian_eom, learned_dynamics(params)))(state)
  return jnp.mean((preds - targets) ** 2)

@jax.jit
def baseline_loss(params, batch, time_step=None):
  state, targets = batch
  preds = jax.vmap(partial(unconstrained_eom, learned_dynamics(params)))(state)
  return jnp.mean((preds - targets) ** 2)


def train(args, model, data):
  global opt_update, get_params, nn_forward_fn
  (nn_forward_fn, init_params) = model
  data = {k: jax.device_put(v) if type(v) is np.ndarray else v for k,v in data.items()}
  time.sleep(2)

  # choose our loss function
  if args.model == 'gln':
    loss = gln_loss
  elif args.model == 'baseline_nn':
    loss = baseline_loss
  else:
    raise(ValueError)

  @jax.jit
  def update_derivative(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, jax.grad(loss)(params, batch, None), opt_state)

  # make an optimizer
  opt_init, opt_update, get_params = optimizers.adam(
    lambda t: jnp.select([t < args.batch_size*(args.num_batches//3),
                          t < args.batch_size*(2*args.num_batches//3),
                          t > args.batch_size*(2*args.num_batches//3)],
                         [args.learn_rate, args.learn_rate/10, args.learn_rate/100]))
  opt_state = opt_init(init_params)

  train_losses, test_losses = [], []
  for iteration in range(args.batch_size*args.num_batches + 1):
    if iteration % args.batch_size == 0:
      params = get_params(opt_state)
      train_loss = loss(params, (data['x'], data['dx']))
      train_losses.append(train_loss)
      test_loss = loss(params, (data['test_x'], data['test_dx']))
      test_losses.append(test_loss)
      if iteration % (args.batch_size*args.test_every) == 0:
        print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
    opt_state = update_derivative(iteration, opt_state, (data['x'], data['dx']))

  params = get_params(opt_state)
  return params, train_losses, test_losses

if __name__ == "__main__":
  args = ObjectView(get_args())
  dblpend.get_dataset(t_span=[0,args.dataset_size], fps=1, samples=1)

  mlp = lagrangian_nns.mlp
  rng = jax.random.PRNGKey(args.seed)
  init_random_params, nn_forward_fn = mlp(args)
  _, init_params = init_random_params(rng, (-1, 4))
  model = (nn_forward_fn, init_params)
  data = dblpend.get_dataset(t_span=[0,args.dataset_size], fps=1, samples=1)

  result = train(args, model, data)