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

"""Task registry."""
import enum
import functools
from typing import Dict, NamedTuple, Optional

import ml_collections
from videogvt.train_lib import cond_utils
from videogvt.train_lib import mask_utils


class SpecialTokens(enum.IntEnum):
  """Vocabulary for special tokens."""

  MASK = -1
  SEP = -2
  TASK_FULL_GENERATION = -3
  TASK_FRAME_PREDICTION = -4
  TASK_FRAME_INTERPOLATION = -5
  TASK_OUTPAINTING = -6
  TASK_INPAINTING = -7
  PAD = -15


class Task(NamedTuple):
  """Data and transformations for a task."""

  task_id: str
  task_token: int
  mask_fn: mask_utils.MaskFn
  cond_fn: Optional[cond_utils.CondFn]
  config: ml_collections.FrozenConfigDict


UNLABELED_DS_TASK_IDS = ['frame_prediction', 'reconstruction']
LABALED_DS_TASK_IDS = ['full_generation'] + UNLABELED_DS_TASK_IDS


def get_full_generation_task(config: ml_collections.ConfigDict, is_train: bool,
                             task_id: str) -> Task:
  """Get full generation task setup."""
  task_token = SpecialTokens.TASK_FULL_GENERATION
  task_config = ml_collections.FrozenConfigDict(config.get(
      task_id, ml_collections.ConfigDict({'class_conditional': True})))
  weight_mode = task_config.get('weight_mode', config.get('weight_mode'))
  block_shape = task_config.get('mask_block_shape', (1, 1, 1))
  if not is_train:
    block_shape = task_config.get('eval_mask_block_shape', block_shape)
  mask_fn = functools.partial(
      mask_utils.random_block_mask,
      condition_segment_id=1,
      weight_mode=weight_mode,
      block_shape=block_shape)
  return Task(task_id, task_token, mask_fn, None, task_config)




def get_latent_frame_prediction_task(config: ml_collections.ConfigDict,
                                     is_train: bool, task_id: str) -> Task:
  """Get latent frame prediction task setup."""
  task_token = SpecialTokens.TASK_FULL_GENERATION
  task_config = ml_collections.FrozenConfigDict(
      config.get(task_id, ml_collections.ConfigDict()))
  weight_mode = task_config.get('weight_mode', config.get('weight_mode'))
  block_shape = task_config.get('mask_block_shape', (1, 1, 1))
  if not is_train:
    block_shape = task_config.get('eval_mask_block_shape', block_shape)
  cond_latent_frames = task_config.get('cond_latent_frames', 1)
  mask_fn = functools.partial(
      mask_utils.random_block_mask,
      condition_mode='cond->cond',
      condition_segment_id=1,
      weight_mode=weight_mode,
      block_shape=block_shape)
  cond_fn = functools.partial(
      cond_utils.latent_frame_prediction_cond,
      cond_latent_frames=cond_latent_frames,
      latent_shape=config.transformer.latent_shape,
  )
  return Task(task_id, task_token, mask_fn, cond_fn, task_config)


def get_frame_prediction_task(config: ml_collections.ConfigDict, is_train: bool,
                              task_id: str) -> Task:
  """Get frame prediction task setup."""
  task_token = SpecialTokens.TASK_FRAME_PREDICTION
  task_config = ml_collections.FrozenConfigDict(
      config.get(task_id, ml_collections.ConfigDict()))
  weight_mode = task_config.get('weight_mode', config.get('weight_mode'))
  block_shape = task_config.get('mask_block_shape', (1, 1, 1))
  if not is_train:
    block_shape = task_config.get('eval_mask_block_shape', block_shape)
  cond_frames = task_config.get('cond_frames', 1)
  cond_padding = task_config.get('cond_padding', 'edge')
  cond_latent_frames = task_config.get('cond_latent_frames', 1)
  condition_mode = task_config.get('condition_mode',
                                   config.get('condition_mode', 'cond->input'))
  mask_fn = functools.partial(
      mask_utils.random_block_mask,
      condition_mode=condition_mode,
      condition_segment_id=1,
      weight_mode=weight_mode,
      block_shape=block_shape)
  cond_fn = functools.partial(
      cond_utils.frame_prediction_cond,
      cond_frames=cond_frames,
      cond_padding=cond_padding,
      cond_latent_frames=cond_latent_frames,
      latent_shape=config.transformer.latent_shape,
      prefix_condition=condition_mode == 'prefix')
  return Task(task_id, task_token, mask_fn, cond_fn, task_config)




def get_frame_interpolation_task(config: ml_collections.ConfigDict,
                                 is_train: bool, task_id: str) -> Task:
  """Get frame interpolation task setup."""
  task_token = SpecialTokens.TASK_FRAME_INTERPOLATION
  task_config = ml_collections.FrozenConfigDict(
      config.get(task_id, ml_collections.ConfigDict()))
  weight_mode = task_config.get('weight_mode', config.get('weight_mode'))
  block_shape = task_config.get('mask_block_shape', (1, 1, 1))
  if not is_train:
    block_shape = task_config.get('eval_mask_block_shape', block_shape)
  cond_frames = task_config.get('cond_frames', 1)
  cond_padding = task_config.get('cond_padding', 'interpolate')
  cond_latent_frames = task_config.get('cond_latent_frames', 1)
  condition_mode = task_config.get('condition_mode',
                                   config.get('condition_mode', 'cond->input'))
  mask_fn = functools.partial(
      mask_utils.random_block_mask,
      condition_mode=condition_mode,
      condition_segment_id=1,
      weight_mode=weight_mode,
      block_shape=block_shape)
  cond_fn = functools.partial(
      cond_utils.frame_interpolation_cond,
      cond_frames=cond_frames,
      cond_padding=cond_padding,
      cond_latent_frames=cond_latent_frames,
      latent_shape=config.transformer.latent_shape)
  return Task(task_id, task_token, mask_fn, cond_fn, task_config)


def get_outpainting_task(config: ml_collections.ConfigDict, is_train: bool,
                         task_id: str) -> Task:
  """Get outpainting task setup."""
  task_token = SpecialTokens.TASK_OUTPAINTING
  task_config = ml_collections.FrozenConfigDict(
      config.get(task_id, ml_collections.ConfigDict()))
  weight_mode = task_config.get('weight_mode', config.get('weight_mode'))
  block_shape = task_config.get('mask_block_shape', (1, 1, 1))
  if not is_train:
    block_shape = task_config.get('eval_mask_block_shape', block_shape)
  cond_region = task_config.get('cond_region', 'quarter_topleft')
  cond_padding = task_config.get('cond_padding', 'edge')
  condition_mode = task_config.get('condition_mode',
                                   config.get('condition_mode', 'cond->input'))
  mask_fn = functools.partial(
      mask_utils.random_block_mask,
      condition_mode=condition_mode,
      condition_segment_id=1,
      weight_mode=weight_mode,
      block_shape=block_shape)
  cond_fn = functools.partial(
      cond_utils.outpainting_cond,
      cond_region=cond_region,
      cond_padding=cond_padding,
      latent_shape=config.transformer.latent_shape)
  return Task(task_id, task_token, mask_fn, cond_fn, task_config)


def get_inpainting_task(config: ml_collections.ConfigDict, is_train: bool,
                        task_id: str) -> Task:
  """Get inpainting task setup."""
  if config.get('bug_patches', {}).get('inpainting_token', False):
    task_token = SpecialTokens.TASK_INPAINTING
  else:
    task_token = SpecialTokens.TASK_OUTPAINTING  # from cl/471393755
  task_config = ml_collections.FrozenConfigDict(
      config.get(task_id, ml_collections.ConfigDict()))
  weight_mode = task_config.get('weight_mode', config.get('weight_mode'))
  block_shape = task_config.get('mask_block_shape', (1, 1, 1))
  if not is_train:
    block_shape = task_config.get('eval_mask_block_shape', block_shape)
  cond_region = task_config.get('cond_region', 'quarter_topleft')
  cond_padding = task_config.get('cond_padding', 'constant')
  condition_mode = task_config.get('condition_mode',
                                   config.get('condition_mode', 'cond->input'))
  mask_fn = functools.partial(
      mask_utils.random_block_mask,
      condition_mode=condition_mode,
      condition_segment_id=1,
      weight_mode=weight_mode,
      block_shape=block_shape)
  cond_fn = functools.partial(
      cond_utils.inpainting_cond,
      cond_region=cond_region,
      cond_padding=cond_padding,
      latent_shape=config.transformer.latent_shape)
  return Task(task_id, task_token, mask_fn, cond_fn, task_config)




def get_structure_task(
    config: ml_collections.ConfigDict, is_train: bool, task_id: str
) -> Task:
  """Get frame from flow task setup."""
  del is_train
  task_config = ml_collections.FrozenConfigDict(
      config.get(task_id, ml_collections.ConfigDict()))
  task_token = SpecialTokens.TASK_STRUCTURE
  mask_fn = functools.partial(mask_utils.all_mask)
  # Currently the cond video is computed directly in token_dumper because the
  # cond_fns only receive the input rgb, but we also need flow. Therefore return
  # None here.
  cond_fn = None
  return Task(task_id, task_token, mask_fn, cond_fn, task_config)


def task_has_audio(task: Task) -> bool:
  if 'audio' in task.task_id:
    return True
  else:
    return False


def task_has_text(task: Task) -> bool:
  del task
  return True




def task_is_structure(task: Task) -> bool:
  return task_id_is_structure(task.task_id)


def get_task_registry(config: ml_collections.ConfigDict,
                      is_train: bool = True) -> Dict[str, Task]:
  """Get the mapping from task name to task."""
  task_ids = config.get('tasks', ('full_generation',))
  tasks = []

  for task_id in task_ids:
    if task_id.startswith('full_generation'):
      tasks.append(get_full_generation_task(config, is_train, task_id))
    elif task_id.startswith('latent_frame_prediction'):
      tasks.append(get_latent_frame_prediction_task(config, is_train, task_id))
    elif task_id.startswith('frame_prediction'):
      tasks.append(get_frame_prediction_task(config, is_train, task_id))
    elif task_id.startswith('frame_interpolation'):
      tasks.append(get_frame_interpolation_task(config, is_train, task_id))
    elif task_id.startswith('outpainting'):
      tasks.append(get_outpainting_task(config, is_train, task_id))
    elif task_id.startswith('inpainting'):
      tasks.append(get_inpainting_task(config, is_train, task_id))
    else:
      raise ValueError(f'Unknown task_id: {task_id}')
  tasks = {task.task_id: task for task in tasks}
  return tasks


def get_dummy_task(task_id: str) -> Task:
  return Task(task_id, SpecialTokens.SEP, mask_utils.all_mask, None,
              ml_collections.FrozenConfigDict())
