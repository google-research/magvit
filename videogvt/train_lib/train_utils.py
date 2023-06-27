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

"""Utility functions for Training."""
import builtins
import copy
import importlib
import itertools
import math
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from absl import logging
import einops
import flax
from flax import jax_utils
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from scenic.dataset_lib import dataset_utils as scenic_dataset_utils
from scenic.dataset_lib import datasets as scenic_datasets
import tensorflow as tf

from videogvt.models import stylegan_discriminator_2d
from videogvt.models import stylegan_discriminator_2plus1d
from videogvt.models import stylegan_discriminator_3d
from videogvt.models import vqvae
from videogvt.train_lib import mask_utils
from videogvt.train_lib import pretrained_model_utils
from videogvt.train_lib import train_state_manager as state_mgr


PyTree = Any
PRNGKey = jnp.ndarray
Batch = Dict[str, jnp.ndarray]
TrainState = state_mgr.TrainState
AllTrainState = state_mgr.AllTrainState

VQ_MODEL_INFO = {
    'VQGAN/2D': {
        'generator_cls': vqvae.VQVAE,
        'discriminator_cls': stylegan_discriminator_2d.StyleGANDiscriminator,
    },
    'VQGAN/2Plus1D': {
        'generator_cls':
            vqvae.VQVAE,
        'discriminator_cls':
            stylegan_discriminator_2plus1d.StyleGANDiscriminator,
    },
    'VQGAN/3D': {
        'generator_cls': vqvae.VQVAE,
        'discriminator_cls': stylegan_discriminator_3d.StyleGANDiscriminator,
    }
}

_CUSTOM_DATASET_TABLE = {
}




def seed_everything(global_seed: int, local_seed: int):
  random.seed(global_seed)
  np.random.seed(global_seed)
  # Different seeds for input pipeline on each device.
  tf.random.set_seed(local_seed)


def get_lax_precision(config):
  lax_precision = config.get('lax_precision', 'default')
  if lax_precision == 'highest':
    return jax.lax.Precision.HIGHEST
  elif lax_precision == 'high':
    return jax.lax.Precision.HIGH
  return jax.lax.Precision.DEFAULT


def get_vq_model(vq_config: ml_collections.ConfigDict) -> Dict[str, nn.Module]:
  """Build VQGAN generator and discriminator models."""
  config = vq_config
  if not ('vqgan' in config or 'model_class' in config or
          'model_type' in config.vqgan):
    raise ValueError('No complete vqgan found in the config.')
  model_info = f'{config.model_class}/{config.vqgan.model_type}'
  if model_info not in VQ_MODEL_INFO:
    raise ValueError(f'{model_info} not known.')
  model_dict = VQ_MODEL_INFO[model_info]
  dtype = get_dtype(config)
  precision = get_lax_precision(config)
  generator = model_dict['generator_cls'](
      config=config, dtype=dtype, precision=precision)
  discriminator = model_dict['discriminator_cls'](config=config, dtype=dtype)
  return {'generator': generator, 'discriminator': discriminator}


def get_dtype(config: ml_collections.ConfigDict):
  if config.dtype == 'bfloat16':
    return jnp.bfloat16
  else:
    return jnp.float32


def flatten_t_dim(x):
  assert x.ndim == 5, 'x should be in shape [bs t h w c]'
  return x.reshape(-1, *x.shape[2:])


def get_additional_data():
  image_model, image_model_state = pretrained_model_utils.get_pretrained_model()
  additional_data = {
      'image_model': image_model,
      'image_model_state': image_model_state,
  }
  return additional_data


def _inputs_add_t_dim(batch: Batch):
  batch['inputs'] = batch['inputs'][:, :, None]  # dev, bs, t, ...
  return batch


def _split_batch_iter(orig_iter: Iterable[Batch], batch_size: int):
  """Split batches from data loader into smaller batch size."""
  num_dev, num_splits = None, None
  for batch in orig_iter:
    if num_dev is None:
      num_dev, orig_dev_bs = jax.tree_util.tree_leaves(batch)[0].shape[:2]
      orig_batch_size = num_dev * orig_dev_bs * jax.process_count()
      assert orig_batch_size % batch_size == 0
      num_splits = orig_batch_size // batch_size
    splitted_batch = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (num_dev, num_splits, -1, *x.shape[2:])),
        batch)
    for i in range(num_splits):
      if 'batch_mask' in splitted_batch and splitted_batch[
          'batch_mask'][:, i].sum() == 0:
        continue
      yield jax.tree_util.tree_map(lambda x: x[:, i], splitted_batch)  # pylint: disable=cell-var-from-loop


def get_dataset(
    config: ml_collections.ConfigDict,
    data_rng: PRNGKey,
    is_train: bool = True,
    *,
    dataset_service_address: Optional[str] = None,
) -> scenic_dataset_utils.Dataset:
  """Get dataset from scenic and transform image datasets."""
  if config.dataset_name == 'unused':
    return scenic_dataset_utils.Dataset(None, None, None, None)
  batch_size = config.batch_size
  dataset_name = config.dataset_name
  if not is_train:
    batch_size = config.get('eval_batch_size', batch_size)
    if config.get('loader_batch_size') is not None:
      batch_size = config.loader_batch_size
    if config.get('eval_dataset_name') is not None:
      # uses a different data_loader for evaluation.
      dataset_name = config.eval_dataset_name
      with config.unlocked():
        # overrides train logic. It's fine as the train_iter is not used here.
        config.dataset_configs.pp_train = config.dataset_configs.pp_eval

  dataset = _get_scenic_dataset(
      config,
      data_rng,
      batch_size,
      dataset_name=dataset_name,
      dataset_service_address=dataset_service_address)
  if dataset.valid_iter and config.dataset_configs.get('num_eval_clips', 1) > 1:
    dataset = dataset._replace(
        valid_iter=_split_batch_iter(dataset.valid_iter, batch_size))
  if dataset.test_iter and (
      config.dataset_configs.get('num_test_clips', 1) > 1 or
      config.dataset_configs.get('do_three_spatial_crops', False)):
    dataset = dataset._replace(
        test_iter=_split_batch_iter(dataset.test_iter, batch_size))
  dataset_type = config.get('dataset_type', 'video')
  if dataset_type == 'video':
    return dataset
  elif dataset_type == 'image':
    s = dataset.meta_data['input_shape']
    dataset.meta_data['input_shape'] = (s[0], 1, *s[1:])
    return dataset._replace(
        train_iter=map(_inputs_add_t_dim, dataset.train_iter)
        if dataset.train_iter else None,
        valid_iter=map(_inputs_add_t_dim, dataset.valid_iter)
        if dataset.valid_iter else None,
        test_iter=map(_inputs_add_t_dim, dataset.test_iter)
        if dataset.test_iter else None)
  else:
    raise NotImplementedError(
        f'Unrecognized dataset type: {dataset_type}')


def _get_scenic_dataset(
    config: ml_collections.ConfigDict,
    data_rng: PRNGKey,
    batch_size: int,
    *,
    start_step: Optional[int] = None,
    num_local_shards: Optional[int] = None,
    dataset_service_address: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_configs: Optional[ml_collections.ConfigDict] = None
) -> scenic_dataset_utils.Dataset:
  """Creates dataset.

  By default, the values in the config file are used.
  However, if the optional `dataset_name` and `dataset_configs` are passed,
    those are used instead.

  Args:
    config: The configuration of the experiment.
    data_rng: Random number generator key to use for the dataset.
    batch_size: Global batch size which applies to all data splits.
    start_step: Start step for GRAIN-backed inpute pipelines.
    num_local_shards: Number of shards for each batch. So (bs, ...) becomes
      (num_local_shards, bs//num_local_shards, ...). If not specified, it will
      be number of local devices.
    dataset_service_address: Used when using the tf.data.experimental.service
    dataset_name: Name of dataset to load, if not reading from the config.
    dataset_configs: Configuration of the dataset, if not reading directly from
      the config.

  Returns:
    A dataset_utils.Dataset object.
  """
  dataset_name = dataset_name or config.dataset_name
  if dataset_name == 'unit_test':
    logging.warn('----->>Using the pseudo dataset. Used only in unit test!')
    return get_pseudo_dataset(config, data_rng, batch_size)

  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  if dataset_name in _CUSTOM_DATASET_TABLE:
    module = _CUSTOM_DATASET_TABLE[dataset_name]
    logging.info('On-demand import of dataset (%s) from module (%s).',
                 dataset_name, module)
    dataset_builder = importlib.import_module(module).get_dataset
  else:
    dataset_builder = scenic_datasets.get_dataset(dataset_name)

  if batch_size % device_count > 0:
    raise ValueError(f'Batch size ({batch_size}) must be divisible by the '
                     f'number of devices ({device_count})')

  local_batch_size = batch_size // jax.process_count()
  device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', local_batch_size)
  logging.info('device_batch_size : %d', device_batch_size)

  shuffle_seed = config.get('shuffle_seed', None)
  if dataset_service_address and shuffle_seed is not None:
    raise ValueError('Using dataset service with a random seed causes each '
                     'worker to produce exactly the same data. Add '
                     'config.shuffle_seed = None to your config if you want '
                     'to run with dataset service.')

  dataset_configs = dataset_configs or config.get('dataset_configs',
                                                  ml_collections.ConfigDict())
  num_local_shards = num_local_shards or jax.local_device_count()


  if dataset_configs.get('expects_start_step'):
    kwargs = {'start_step': start_step}
  else:
    kwargs = {}

  dataset = dataset_builder(
      batch_size=local_batch_size,
      eval_batch_size=local_batch_size,
      num_shards=num_local_shards,
      dtype_str=config.data_dtype_str,
      rng=data_rng,
      shuffle_seed=shuffle_seed,
      dataset_configs=dataset_configs,
      dataset_service_address=dataset_service_address,
      **kwargs)

  return dataset


def save_checkpoint(workdir: str,
                    train_state: AllTrainState,
                    max_to_keep: int = 3,
                    overwrite: bool = False):
  """Saves a checkpoint.

  First syncs the model state across replicas, then it unreplicates it by taking
  the train state of the first replica and saves it as a checkpoint.

  Args:
    workdir: Experiment directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    max_to_keep: The number of checkpoints to keep.
    overwrite: Overwrite existing checkpoint  if a checkpoint
      at the current or a later step already exits (default: False).
  """
  # Get train state from the first replica.
  checkpoint_state = jax.device_get(jax_utils.unreplicate(train_state))
  checkpoints.save_checkpoint_multiprocess(
      workdir,
      checkpoint_state,
      int(checkpoint_state.global_step),
      overwrite=overwrite,
      keep=max_to_keep)


def flattened_traversal(fn):
  """Utility function to create masks for optax.

  Adapted from:
  https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md

  Args:
    fn: a filter function e.g., lambda path, _: path[-1] == 'bias'

  Returns:
    The function to produce the mask.
  """
  def mask(data):
    flat = flax.traverse_util.flatten_dict(data)
    return flax.traverse_util.unflatten_dict(
        {k: fn(k, v) for k, v in flat.items()})

  return mask


def apply_on_global_batch(fn, **kwargs):
  """Transforms a fn to operate on global batch with (num_dev, dev_bs, ...)."""

  def new_fn(*input_seq):
    input_set = []
    for an_input in input_seq:
      num_dev, dev_bs = jax.tree_util.tree_leaves(an_input)[0].shape[:2]
      inputs = jax.tree_util.tree_map(
          lambda x: x.reshape(-1, *x.shape[2:]), an_input)
      input_set.append(inputs)
    outputs = fn(*input_set, **kwargs)
    outputs = jax.tree_util.tree_map(
        lambda x: x.reshape(num_dev, dev_bs, *x.shape[1:]), outputs)
    return outputs

  return new_fn


def all_gather_cpu(tree: PyTree) -> PyTree:
  """Apply multi-host all_gather operation on a tree of CPU arrays.

  Args:
    tree: Input tree containing np.ndarray in shape of (num_dev, dev_bs, ...).
  Returns:
    Output tree in the same structure as input with shape (global_bs, ...),
    where global_bs = num_host * num_dev * dev_bs.
  """
  if jax.process_count() == 1:
    return jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), tree)
  tree = jax.tree_util.tree_map(jnp.asarray, tree)  # (num_dev, dev_bs, ...)
  tree = jax.pmap(  # (num_dev, global_bs, ...)
      lambda x: jax.lax.all_gather(x, axis_name='device', tiled=True),
      axis_name='device',
  )(tree)
  tree = jax_utils.unreplicate(tree)  # (global_bs, ...)
  tree = jax.tree_util.tree_map(np.asarray, tree)
  return tree


def _convert_pre_linen(restored_model_state: Mapping[str, Any]):
  if restored_model_state:
    restored_model_state = checkpoints.convert_pre_linen(
        flax.traverse_util.unflatten_dict({
            tuple(k.split('/')[1:]): v for k, v in restored_model_state.items()
        }))
  return restored_model_state


def sum_model_params(
    params: Union[Dict[str, Any], flax.core.FrozenDict]) -> float:
  """Absolute sum of the parameters useful for checking finetuning."""
  all_leaves = jax.tree_util.tree_leaves(params)
  params_sum = []
  for leaf in all_leaves:
    params_sum.append(jnp.sum(jnp.abs(leaf)))
  return np.sum(params_sum, axis=None)


def check_training_step(steps_per_epoch_from_dataset: int,
                        config: ml_collections.ConfigDict):
  """Checks whether the step in learning rate and config matches."""
  lr_configs = config.lr_configs
  steps_per_cycle = lr_configs.get('steps_per_cycle')
  total_steps = lr_configs.get('total_steps')
  assert not steps_per_cycle or not total_steps, (
      f'Do not set both steps_per_cycle {total_steps} and '
      f'total_steps{total_steps}.')

  total_steps_in_lr = total_steps or steps_per_cycle
  assert total_steps_in_lr is not None and total_steps_in_lr > 0, (
      'Must define total_steps or steps_per_cycle.')
  if config.get('lr_configs.steps_per_epoch'):
    given_steps_per_epoch = config.lr_configs.steps_per_epoch
    if steps_per_epoch_from_dataset != given_steps_per_epoch:
      logging.warning(('Mismatched step_per_epoch: given=%d vs actual=%d '
                       'batch_size=%d'), given_steps_per_epoch,
                      steps_per_epoch_from_dataset, config.batch_size)




def make_grid(samples, show_num=64):
  """Tile images to an image grid."""
  batch_size, height, width, c = samples.shape
  if batch_size < show_num:
    logging.info('show_num is cut by the small batch size to %d', batch_size)
    show_num = batch_size
  h_num = int(math.sqrt(show_num))
  w_num = int(show_num / h_num)
  grid_num = h_num * w_num

  samples = samples[0:grid_num]
  samples = samples.reshape(h_num, w_num, height, width, c)
  samples = samples.swapaxes(1, 2)
  samples = samples.reshape(height * h_num, width * w_num, c)
  return samples


def draw_frames_side_by_side(*video_batches: jnp.ndarray,
                             show_num: int = 4,
                             border: int = 1):
  """Draws the original and generated frames side-by-side."""
  assert all([v.ndim == 5 for v in video_batches]), 'Incorrect input format.'
  swap_ins = 'device_bs t h w c -> device_bs h (t w) c'
  vs = [einops.rearrange(v, swap_ins) for v in video_batches]
  v = np.concatenate(vs, 1)
  if border > 0:
    v = np.pad(v, ((0, 0), (border, border), (border, border), (0, 0)))
  return make_grid(v, show_num)


def draw_videos_grid(video_batch: np.ndarray,
                     show_num: int = 4,
                     border: int = 1):
  """Draws the videos as a grid."""
  assert video_batch.ndim == 5, 'Incorrect input format.'
  batch_size, time, height, width, c = video_batch.shape
  if batch_size < show_num:
    logging.info('show_num is cut by the small batch size to %d', batch_size)
    show_num = batch_size
  h_num = max(1, int(math.sqrt(show_num * width / height)))
  w_num = int(show_num / h_num)
  grid_num = h_num * w_num

  samples = video_batch[0:grid_num]
  if border > 0:
    samples = np.pad(samples, ((0, 0), (0, 0), (border, border),
                               (border, border), (0, 0)))
    height, width = height + 2 * border, width + 2 * border
  samples = samples.reshape(h_num, w_num, time, height, width, c)
  samples = einops.rearrange(samples, 'bh bw t h w c -> t (bh h) (bw w) c')
  samples = samples.reshape(time, height * h_num, width * w_num, c)
  return samples


def draw_videos_side_by_side(*video_batches: jnp.ndarray,
                             show_num: int = 4):
  """Draws the original and generated videos side-by-side."""
  assert all([v.ndim == 5 for v in video_batches]), 'Incorrect input format.'
  v = np.concatenate(video_batches, 2)
  return draw_videos_grid(v, show_num)


def get_pseudo_dataset(config: ml_collections.ConfigDict,
                       rng: jnp.ndarray,
                       batch_size: Optional[int] = None):
  """For the unit tests to avoid the cns access issues."""
  if not batch_size:
    batch_size = config.batch_size
  train_ds = []
  num_of_devices = jax.local_device_count()
  local_bs = batch_size // num_of_devices
  image_size = config.image_size
  if isinstance(image_size, int):
    height = width = image_size
  else:
    height, width = image_size
  num_frames = config.dataset_configs.num_frames
  num_classes = config.dataset_configs.num_classes
  num_batches = config.debug_num_batches

  for batch_idx in range(num_batches):
    # (batch, T, H, W, C)
    rng, video_rng, label_rng = jax.random.split(rng, 3)
    videos = jax.random.uniform(
        video_rng,
        shape=(num_of_devices, local_bs, num_frames, height, width,
               3)).astype(jnp.float32)
    label = jax.random.choice(label_rng, num_classes,
                              [num_of_devices, local_bs, 1])
    # Media ids should be an array of lists.
    if config.dataset_configs.get('include_video_id', False):
      media_ids = []
      for d_idx in range(num_of_devices):
        lb_ids = [
            f'id_{batch_idx*batch_size + d_idx*local_bs + lb_idx}'.encode()
            for lb_idx in range(local_bs)
        ]
        media_ids.append(lb_ids)
      media_ids = np.array(media_ids)
    else:
      media_ids = None

    batch_mask = jnp.ones_like(label[..., 0])
    # Simulate incomplete last batch that got padded.
    if batch_idx == num_batches - 1 and config.get(
        'debug_last_batch_padded', False
    ):
      assert (
          local_bs >= 2
      ), 'local_bs should be >= 2 for debug_last_batch_padded option'
      batch_mask = batch_mask.at[:, local_bs - 1:].set(0)
    sample = dict(inputs=videos, label=label, batch_mask=batch_mask)
    if media_ids is not None:
      sample['vid'] = media_ids
    train_ds.append(sample)
  meta_data = dict(
      input_dtype=jnp.float32,
      input_shape=[-1, num_frames, height, width, 3],
      num_classes=num_classes,
      num_train_examples=num_batches * config.batch_size,
      num_eval_examples=num_batches * config.batch_size,
      num_test_examples=num_batches * config.batch_size,
      target_is_onehot=False,
      is_pseudo=True)

  dataset = scenic_dataset_utils.Dataset(
      train_iter=itertools.cycle(train_ds),
      valid_iter=itertools.cycle(train_ds),
      test_iter=itertools.cycle(train_ds),
      meta_data=meta_data)
  return dataset


def restore_checkpoint(checkpoint_path: str,
                       train_state: state_mgr.TrainState,
                       step: Optional[int] = None,
                       is_legacy_checkpoint: Optional[bool] = False):
  """Restores a legacy checkpoint.

  Args:
    checkpoint_path: Directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    step: Step number to load or None to load the latest. If specified,
      checkpoint_path must be a directory.
    is_legacy_checkpoint: whether in the legacy checkpoint format.

  Returns:
    Training state and an int which is the current step.
  """
  if is_legacy_checkpoint:
    train_state = restore_legacy_checkpoint(checkpoint_path, train_state, step)
  else:
    train_state = checkpoints.restore_checkpoint(
        checkpoint_path, train_state, step
    )
  return train_state


def restore_finetune_checkpoint(
    checkpoint_path: str,
    train_state: state_mgr.VQGANTrainState,
    step: Optional[int] = None,
    not_restore_g_params_keys: Optional[List[str]] = None,
):
  """Restores a checkpoint for VQ finetuning.

  Args:
    checkpoint_path: Directory for saving the checkpoint.
    train_state: the target TrainState.
    step: Step number to load or None to load the latest. If specified,
      checkpoint_path must be a directory.
    not_restore_g_params_keys: List of g_params keys not to restore from the
      checkpoint. Default (None) loads every key.

  Returns:
    Training state and an int which is the current step.
  """
  restored_train_state = checkpoints.restore_checkpoint(
      ckpt_dir=checkpoint_path, target=None, step=step
  )

  g_params = train_state.g_params
  restored_g_params = restored_train_state['g_params']
  for k in restored_g_params.keys():
    if not_restore_g_params_keys and k not in not_restore_g_params_keys:
      g_params[k] = restored_g_params[k]

  train_state = train_state.replace(
      # global_step=restored_train_state['global_step'],  # step starts from 0
      g_model_state=restored_train_state.get('g_model_state'),
      d_model_state=restored_train_state.get('d_model_state'),
      rng=restored_train_state.get('rng'),
      metadata=restored_train_state.get('metadata'),
      g_params=restored_train_state['g_params'],
      d_params=restored_train_state['d_params'],
      ema_params=restored_train_state['ema_params'],
  )
  return train_state


def restore_legacy_checkpoint(
    checkpoint_path: str,
    train_state: state_mgr.TrainState,
    step: Optional[int] = None,
):
  """Restores a legacy checkpoint.

  Args:
    checkpoint_path: Directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    step: Step number to load or None to load latest. If specified,
      checkpoint_path must be a directory.

  Returns:
    Training state and an int which is the current step.
  """
  restored_train_state = checkpoints.restore_checkpoint(checkpoint_path, None,
                                                        step)
  if restored_train_state is None:
    raise ValueError('No checkpoint for the pretrained model is found in: '
                     f'{checkpoint_path}')
  restored_train_state_type = state_mgr.check_train_state_type(
      restored_train_state)
  if isinstance(train_state, restored_train_state_type):
    if (restored_train_state_type == state_mgr.ScenicTrainState) and (
        'metadata' not in restored_train_state):
      restored_train_state.pop('accum_train_time', None)
    if restored_train_state.get('metadata') is None:
      restored_train_state['metadata'] = {}
    restored_train_state['metadata'].setdefault('lecam_ema_real', None)
    restored_train_state['metadata'].setdefault('lecam_ema_fake', None)
    train_state = flax.serialization.from_state_dict(train_state,
                                                     restored_train_state)
  elif restored_train_state_type == state_mgr.VQGANTrainStateDeprecated:
    assert isinstance(train_state, state_mgr.VQGANTrainState)
    logging.warn('Optimizer state cannot be restored for %s',
                 restored_train_state_type)
    train_state = train_state.replace(  # pytype: disable=attribute-error
        global_step=restored_train_state['global_step'],
        g_model_state=restored_train_state.get('generator_state'),
        d_model_state=restored_train_state.get('discriminator_state'),
        rng=restored_train_state.get('rng'),
        metadata=restored_train_state.get('metadata'),
        g_params=restored_train_state['g_optimizer']['target'],
        d_params=restored_train_state['d_optimizer']['target'],
        ema_params=restored_train_state['ema_params'],
    )
  else:
    raise ValueError(f'Train_state {type(train_state)} is not supported.')
  return train_state


def pool_overlapping_segments(x: np.ndarray,
                              overlapping_frames: int = 4,
                              alphas: Optional[Any] = None):
  """weighted pools the output pixels or the intermediate embeddings.

  Args:
    x: video pixels or embeddings in [#segments t h w c].
    overlapping_frames: overlapping frames to pool.
    alphas: A scalar of a 1d array of len overlapping_frames.

  Returns:
    Features pooled by overlapping segments.
  """
  assert x.ndim == 5, 'x should be in shape [bs t h w c]'
  _, t = x.shape[0], x.shape[1]
  assert overlapping_frames > 0, f'Bad overlapping_frames={overlapping_frames}'
  assert overlapping_frames <= t // 2, 'Case not supported yet.'
  if alphas is None:  # linear interpolation
    alphas = np.arange(0., 1., step=1./overlapping_frames)
    alphas = alphas[:, None, None, None]

  step_size = t-overlapping_frames

  ret = []
  first_segment = x[0, 0: (t-overlapping_frames), ...]
  last_segment = x[-1, -overlapping_frames:]
  for i in range(1, x.shape[0]):
    cur_segment = (1 - alphas) * x[i - 1, -overlapping_frames:] + alphas * x[
        i, 0:overlapping_frames]
    ret.append(cur_segment)
    if step_size > overlapping_frames:
      ret.append(x[i, overlapping_frames:step_size])
  ret = np.concatenate(ret)
  ret = np.reshape(ret, [-1, *x.shape[2:]])
  ret = np.concatenate((first_segment, ret, last_segment))
  num_pad_frames = int(np.ceil(ret.shape[0] / t)*t-ret.shape[0])
  if num_pad_frames > 0:
    ret = np.concatenate((ret, np.zeros([num_pad_frames, *x.shape[2:]])))
  ret = np.reshape(ret, [-1, *x.shape[1:]])
  return ret




def create_sstable_builder(
    result_dir: str, *, num_shards: int = 8, sort: bool = False
):
  """Creates a context manager for the sstable writer."""
  shard_spec = os.path.join(result_dir, f'p{jax.process_index()}@{num_shards}')
  shard_paths = gfile.GenerateShardedFilenames(shard_spec)
  # Remove partial files from previous run.
  for path in shard_paths:
    if gfile.Exists(path):
      gfile.Remove(path)
  sharder = sstable_sharder.SSTable_SharderFactory.New(
      'SSTable_KeyFingerprintSharder')
  if sort:
    return sharded_builder.SortingShardedBuilder(shard_paths, sharder)
  else:
    return sharded_builder.ShardedBuilder(shard_paths, sharder)


class LongVideoIterable:
  """Simple iterable class for a long video in memory."""

  def __init__(self,
               video: np.ndarray,
               batch_size: int = 32,
               temporal_window: int = 16,
               overlapping_frames: int = 0):
    assert video.ndim == 4  # [frames, h, w, c]
    assert batch_size > 0  # batch_size % jax.device_count() == 0
    self._batch_size = batch_size

    self._video = video
    window_size = temporal_window
    self._window_size = window_size
    self._overlapping_frames = overlapping_frames

    # self._index = 0
    # frame_per_batch = temporal_window * batch_size
    frame_per_batch = window_size + (batch_size - 1) * (
        window_size - overlapping_frames)
    self._frame_per_batch = frame_per_batch
    num_windows = (video.shape[0] - window_size) // (window_size -
                                                     overlapping_frames) + 1
    self._num_windows = num_windows
    # drops the last frames
    self._max_iter = num_windows // batch_size  # drops the remaining frames.
    # logging.info('num_windows=%d, max_iter=%d', self._num_windows,
    #              self._max_iter)

  def __iter__(self):
    self._cur_iter = 0
    return self

  def __next__(self):
    if self._cur_iter < self._max_iter:
      frame_pre_batch = self._frame_per_batch
      j = self._cur_iter * (frame_pre_batch - self._overlapping_frames
                           )  # start index
      # logging.info('start_index=%d, _cur_iter=%d', j, self._cur_iter)

      if self._overlapping_frames == 0:
        result = self._video[j:(j+frame_pre_batch)]
      else:
        result = []
        for _ in range(self._batch_size):
          result.append(self._video[j:(j+self._window_size)])
          j += self._window_size - self._overlapping_frames
        result = np.array(result)
      result = result.reshape([-1, self._window_size, *self._video.shape[1:]])
      self._cur_iter += 1
      return dict(inputs=result)
    else:
      raise StopIteration
