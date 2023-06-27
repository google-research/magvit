# Copyright 2023 The videogvt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lib for the VQGAN trainer."""
import functools
from typing import Callable, Dict, Any

from absl import logging
from clu import metrics
import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import optax
from tensorflow.io import gfile
from videogvt.train_lib import train_state_manager

TrainState = train_state_manager.VQGANTrainState
TrainStateDeprecated = train_state_manager.VQGANTrainStateDeprecated


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """metrics for vqgan."""
  d_loss: metrics.Average.from_output('d_loss')
  g_loss: metrics.Average.from_output('g_loss')
  quantizer_loss: metrics.Average.from_output('quantizer_loss')
  reconstruction_loss: metrics.Average.from_output('reconstruction_loss')
  g_adversarial_loss: metrics.Average.from_output('g_adversarial_loss')
  perceptual_loss: metrics.Average.from_output('perceptual_loss')
  d_adversarial_loss: metrics.Average.from_output('d_adversarial_loss')
  grad_penalty: metrics.Average.from_output('grad_penalty')
  logit_laplace_loss: metrics.Average.from_output('logit_laplace_loss')
  lecam_loss: metrics.Average.from_output('lecam_loss')


def zero_grads():
  """Zeros gradient function."""
  # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
  def init_fn(_):
    return ()

  def update_fn(updates, state, params=None):
    del state, params
    return jax.tree_map(jnp.zeros_like, updates), ()

  return optax.GradientTransformation(init_fn, update_fn)


def compute_ema_params(
    ema_params: Any,
    new_params: Any,
    config: ml_collections.ConfigDict,
):
  """Computes the ema parmameters."""
  ema_decay = config.polyak_decay
  ema_fn = lambda emap, newp: emap * ema_decay + (1 - ema_decay) * newp
  if isinstance(new_params, Dict) and isinstance(
      ema_params, flax.core.FrozenDict
  ):
    # to match ema_params which is in frozen dict.
    new_params = flax.core.freeze(new_params)
  new_ema_params = jax.tree_util.tree_map(ema_fn, ema_params, new_params)
  return new_ema_params


def get_optimizer(
    g_learning_rate_fn: Callable[[int], float],
    d_learning_rate_fn: Callable[[int], float],
    config: ml_collections.ConfigDict,
) -> Dict[str, optax.GradientTransformation]:
  """Constructs the optimizer from the given HParams."""
  weight_decay = config.optimizer.get('weight_decay', 0.0)
  g_tx = optax.adamw(  # pytype: disable=wrong-arg-types  # numpy-scalars
      learning_rate=g_learning_rate_fn,
      b1=config.optimizer.beta1,
      b2=config.optimizer.beta2,
      weight_decay=weight_decay)
  d_tx = optax.adamw(  # pytype: disable=wrong-arg-types  # numpy-scalars
      learning_rate=d_learning_rate_fn,
      b1=config.optimizer.beta1,
      b2=config.optimizer.beta2,
      weight_decay=weight_decay)
  if config.get('vqgan.finetune_decoder'):
    logging.info('Finetuning the VQVAE decoder only.')
    param_labels = {'decoder': 'adamw', 'encoder': 'zero', 'quantizer': 'zero'}
    g_tx = optax.multi_transform(
        {'adamw': g_tx,
         'zero': zero_grads()},
        param_labels)
  tx_dict = dict(generator=g_tx, discriminator=d_tx)
  return tx_dict


def sync_model_state_across_replicas(state: TrainState) -> TrainState:
  """Sync the model_state (simple averaging) across replicas."""
  # TODO(): We simply do "mean" here and this doesn't work with
  #   statistics like variance. (check the discussion in Flax for fixing this).
  pmap_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
  if jax.tree_util.tree_leaves(state.g_model_state):
    g_model_state = pmap_mean(state.g_model_state)
  else:
    g_model_state = state.g_model_state
  if jax.tree_util.tree_leaves(state.d_model_state):
    d_model_state = pmap_mean(state.d_model_state)
  else:
    d_model_state = state.d_model_state
  return state.replace(
      g_model_state=g_model_state,
      d_model_state=d_model_state)


def inflate_2d_to_3d(tree_2d, tree_3d, *, mode='central'):
  """Inflate params of 2D CNN to 3D CNN.

  Note: If tree_2d is not a frozen dict and tree_3d is a regular dict,
  tree_2d is unfrozen before inflation.

  Args:
    tree_2d: The tree of 2D params.
    tree_3d: The tree of 3D params.
    mode: One of 'average' or 'central', the mode to use for parameter
      inflation.

  Returns:
    The inflated 3D tree.
  """
  assert mode in ['average', 'central'], f'Unknown inflation mode {mode}'

  if isinstance(tree_2d, flax.core.FrozenDict) and isinstance(tree_3d, dict):
    tree_2d = flax.core.unfreeze(tree_2d)

  def inflate_param(param_2d, param_3d):
    if param_2d.shape == param_3d.shape:
      return param_2d
    if param_2d.shape == param_3d.shape[1:]:
      if mode == 'average':
        param_2d_replicated = jnp.broadcast_to(param_2d[None], param_3d.shape)
        return param_2d_replicated / param_3d.shape[0]
      elif mode == 'central':
        param_3d_zero = jnp.zeros_like(param_3d)
        mid = param_3d.shape[0] // 2
        return param_3d_zero.at[mid].set(param_2d)
    logging.warn('Unable to inflate from %s to %s, left as is.', param_2d.shape,
                 param_3d.shape)
    return param_3d

  return jax.tree_util.tree_map(inflate_param, tree_2d, tree_3d)


def _load_embedding(path, start_idx, end_idx):
  with gfile.GFile(path, 'rb') as f:
    embedding = flax.serialization.from_bytes(None, f.read()).T
  return embedding[start_idx:end_idx]


def _update_embedding(tree, embedding):
  assert embedding.shape == tree['quantizer']['codebook'].shape
  tree = flax.core.unfreeze(tree)
  tree['quantizer']['codebook'] = embedding
  tree = flax.core.freeze(tree)
  return tree


def init_from_pretrained_checkpoint(checkpoint_path: str,
                                    train_state: TrainState,
                                    config: ml_collections.ConfigDict):
  """Initialize the train state with a pretrained checkpoint.

  First restores the checkpoint, then inflates the parameters if requested,
  finally build a new train state with updated parameters.

  Args:
    checkpoint_path: Directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    config: Model config.

  Returns:
    Training state.
  """
  assert train_state is not None, 'train_state cannot be None.'
  restored_train_state = checkpoints.restore_checkpoint(checkpoint_path, None)
  if restored_train_state is None:
    raise ValueError('No checkpoint for the pretrained model is found in: '
                     f'{checkpoint_path}')
  restored_train_state_type = train_state_manager.check_train_state_type(
      restored_train_state)
  restored_train_state = flax.core.freeze(restored_train_state)
  if restored_train_state_type == TrainStateDeprecated:
    # deprecated train_state format
    g_restored_params = restored_train_state['g_optimizer']['target']
    g_restored_model_state = restored_train_state.get('generator_state')
    d_restored_params = restored_train_state['d_optimizer']['target']
    d_restored_model_state = restored_train_state.get('discriminator_state')
  elif restored_train_state_type == TrainState:
    # model trained using new optax optimizers.
    g_restored_params = restored_train_state['g_params']
    g_restored_model_state = restored_train_state.get('g_model_state')
    d_restored_params = restored_train_state['d_params']
    d_restored_model_state = restored_train_state.get('d_model_state')
  else:
    raise ValueError('restored_train_state is not supported.')
  ema_restored_params = restored_train_state['ema_params']

  inflation = config.init_from.get('inflation')
  if inflation is not None:
    if '/' in inflation:
      direction, mode = inflation.split('/')
    else:
      direction, mode = inflation, 'central'
    if direction == '2d->3d':
      inflate_fn = functools.partial(inflate_2d_to_3d, mode=mode)
    else:
      raise NotImplementedError(f'Inflation {inflation}')
    inflate_fn = jax.jit(inflate_fn, backend='cpu')
    g_restored_params = inflate_fn(g_restored_params,
                                   train_state.g_params)
    g_restored_model_state = inflate_fn(
        g_restored_model_state or flax.core.FrozenDict(),
        train_state.g_model_state)
    d_restored_params = inflate_fn(d_restored_params,
                                   train_state.d_params)
    d_restored_model_state = inflate_fn(
        d_restored_model_state or flax.core.FrozenDict(),
        train_state.d_model_state)
    ema_restored_params = inflate_fn(ema_restored_params,
                                     train_state.ema_params)
  if config.init_from.get('embedding_path') is not None:
    embedding = _load_embedding(
        config.init_from.embedding_path,
        config.init_from.get('embedding_start'),
        config.init_from.get('embedding_end'),
    )
    g_restored_params = _update_embedding(g_restored_params, embedding)
    ema_restored_params = _update_embedding(ema_restored_params, embedding)
  # pytype: disable=attribute-error
  new_train_state = train_state.replace(
      g_params=g_restored_params,
      d_params=d_restored_params,
      g_model_state=g_restored_model_state,
      d_model_state=d_restored_model_state,
      ema_params=ema_restored_params)

  # pytype: enable=attribute-error
  return new_train_state
