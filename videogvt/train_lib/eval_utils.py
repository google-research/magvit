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

"""Utils for evaluation metrics.

Selection priority of the "num_examples" for eval:
1. config.eval.num_examples
2. ds_info['num_examples']
3. dataset.meta_data['num_eval_examples']
"""

import collections
import concurrent.futures
import functools
import itertools
import json
import math
import os
import pickle
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from absl import logging
from clu import periodic_actions
from clu import platform
import jax
import jax.numpy as jnp
import mediapy
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.io import gfile
from videogvt.train_lib import frechet_distance
from videogvt.train_lib import image_quality_metrics
from videogvt.train_lib import inception_score
from videogvt.train_lib import maskgvt_train_lib
from videogvt.train_lib import metrics_lib
from videogvt.train_lib import task_manager
from videogvt.train_lib import task_registry
from videogvt.train_lib import train_state_manager
from videogvt.train_lib import train_utils


Batch = train_utils.Batch
EvalFeatureDict = metrics_lib.EvalFeatureDict
EvalFeatureDictCPU = metrics_lib.EvalFeatureDictCPU
EvalMetricDict = Dict[str, Union[int, float]]
TrainState = train_state_manager.TrainState
TaskManager = task_manager.CustomTaskManager

UNLABELED_DS_TASK_IDS = task_registry.UNLABELED_DS_TASK_IDS
LABELED_DS_TASK_IDS = task_registry.LABALED_DS_TASK_IDS
# DEFAULT_TASK_IDS = UNLABELED_DS_TASK_IDS

EVAL_DATASET_INFO = {
    'ucf101': {
        'scenic_name': 'ucf101',
        'split': 'validation',
        'num_examples': 3_783,
        'num_repeats': 4,  # num_repeats to compute FVD
        'supported_tasks': LABELED_DS_TASK_IDS,
    },
    'ucf101_train': {
        'scenic_name': 'ucf101',
        'split': 'train',
        'num_examples': 9_537,
        'num_repeats': 4,
        'supported_tasks': LABELED_DS_TASK_IDS,
    },
    'bair': {
        'scenic_name': 'bair',
        'split': 'validation',
        'num_examples': 256 * 100,
        'num_repeats': 1,
        'supported_tasks': UNLABELED_DS_TASK_IDS,
    },
    'bair_train': {
        'scenic_name': 'bair',
        'split': 'train',
        'num_examples': 43_264,
        'num_repeats': 1,
        'supported_tasks': UNLABELED_DS_TASK_IDS,
    },
    'ssv2': {
        'scenic_name': 'ssv2',
        'split': 'validation',
        'num_examples': 50_000,
        'num_repeats': 1,
        'supported_tasks': LABELED_DS_TASK_IDS,
    },
}

YOUTUBE_RESOLUTION = {
    '240p': (426, 240),
    '360p': (640, 360),
    '480p': (854, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '1440p': (2560, 1440),
    '2160p': (3840, 2160),
}

gather_outputs_with_mask = metrics_lib.gather_outputs_with_mask
run_lpips_models = functools.partial(
    image_quality_metrics.run_models, is_tf_function=True)


def get_allowed_dataset_names(dataset_name: str,
                              task_id: str) -> Tuple[str, ...]:
  """(dataset_name, task) maps to all possible datasets."""
  # TODO(roadjiang): refactor this function.
  if dataset_name == 'unit_test':
    return ('unit_test',)
  ds_names = [
      t for t in EVAL_DATASET_INFO.keys()
      if t.startswith(dataset_name)
  ]
  ret = []
  for ds_name in ds_names:
    supported_tasks = EVAL_DATASET_INFO[ds_name].get('supported_tasks',
                                                     UNLABELED_DS_TASK_IDS)
    if task_id in supported_tasks:
      ret.append(ds_name)
  return tuple(ret)


def split_batch_iter(orig_iter: Iterable[train_utils.Batch], num_splits: int):
  dev_bs = None
  for batch in orig_iter:
    if dev_bs is None:
      orig_dev_bs = jax.tree_util.tree_leaves(batch)[0].shape[1]
      assert orig_dev_bs % num_splits == 0
      dev_bs = orig_dev_bs // num_splits
    for i in range(num_splits):
      yield jax.tree_util.tree_map(lambda x: x[:, i * dev_bs:(i + 1) * dev_bs],  # pylint: disable=cell-var-from-loop
                                   batch)


def get_eval_data_iters(dataset: dataset_utils.Dataset,
                        config: ml_collections.ConfigDict,
                        is_final_eval: bool = False):
  """Get data iterators with number of steps for evaluation."""
  data_iters = collections.OrderedDict()
  batch_size_multiplier = 1
  for data_split in config.eval.data_splits.split(','):
    # Get iterator for the split with total examples.
    if data_split == 'train':
      num_examples_per_epoch = dataset.meta_data['num_train_examples']
      data_iter = dataset.train_iter
    elif data_split == 'validation':
      num_examples_per_epoch = dataset.meta_data['num_eval_examples']
      if config.dataset_configs.get('num_eval_clips', 1) > 1:
        batch_size_multiplier *= config.dataset_configs['num_eval_clips']
      data_iter = dataset.valid_iter
    elif data_split == 'test':
      num_examples_per_epoch = dataset.meta_data['num_test_examples']
      if config.dataset_configs.get('num_test_clips', 1) > 1:
        batch_size_multiplier *= config.dataset_configs['num_test_clips']
      if config.dataset_configs.get('do_three_spatial_crops', False):
        batch_size_multiplier *= 3
      data_iter = dataset.test_iter
    else:
      raise ValueError(f'Invalid data split: {data_split}')

    if config.eval.get('num_examples') is not None:
      total_num_examples = config.eval.num_examples
    else:
      total_num_examples = num_examples_per_epoch

    num_repeats = 1
    # Run more at final-step evaluation
    if is_final_eval:
      num_repeats = config.eval.get('final_num_repeats', 1)
      num_example_multiplier = config.eval.get('final_num_example_multiplier',
                                               1)
      total_num_examples *= num_repeats * num_example_multiplier

    # Add a few steps to account for the partial batch at epoch end
    batch_size = batch_size_multiplier * config.eval_batch_size
    if num_examples_per_epoch > 0:
      valid_fraction = num_examples_per_epoch / (
          math.ceil(num_examples_per_epoch / batch_size) * batch_size)
    else:
      valid_fraction = 1.
    num_steps = int(
        math.ceil(total_num_examples / config.eval_batch_size / valid_fraction))
    data_iters[data_split] = (data_iter, num_steps, total_num_examples,
                              num_repeats)
  return data_iters


def get_train_config(
    config: ml_collections.ConfigDict,
    override_prefix: Optional[Tuple[str, ...]] = None
) -> Tuple[ml_collections.ConfigDict, str]:
  """Gets the train_config information from global config.


  Args:
    config: Configurations of the experiment.
    override_prefix: tuple of string prefix to override the values in the
      train_config or None if nothing is overridden

  Returns:
    The train_config with updated values for the specified prefixes.
  """
  assert 'eval_from' in config, 'eval_from is missing in the config.'
  train_config = config
  ckpt_path = ''
  if config.eval_from.get('checkpoint_path') is not None:
    # uses the user-provided dir to override.
    ckpt_path = config.eval_from.checkpoint_path
  return train_config, ckpt_path


def _repeat_last(iterator: Iterable[Any], num_repeats: int):
  element = None
  for element in iterator:
    yield element
  if element is None:
    return
  for _ in range(num_repeats - 1):
    yield element


def get_eval_jobs(workdir: str,
                  config: ml_collections.ConfigDict,
                  final_num_repeats: int = 1,
                  *,
                  custom_outputs: bool = False):
  """Get evaluation jobs from checkpoint manager.

  Two possible cases:
  1. When jobs run train and eval in parallel, we knew the ckpt_dir=workdir,
    and config.
  2. For a single eval job, the user needs to specify config.eval_from and
    it will look up the information for the train_config.

  Args:
    workdir: Experiment directory for saving the checkpoint.
    config: Configurations of the experiments used in the trainer.
    final_num_repeats: Number of repeats for the last checkpoint.
    custom_outputs: Whether to load custom outputs.

  Returns:
    The train_config with updated values for the specified prefixes,
      ckpt_manager, and ckpt_list.
  """
  if 'eval_from' in config:
    # separate eval job, gets info from the train experiment
    override_prefix = ('eval', 'data', 'log', 'batch',
                       'sampling') + config.eval_from.get(
                           'override_prefix', tuple())
    train_config, ckpt_path = get_train_config(config, override_prefix)
    if not gfile.isdir(ckpt_path):
      ckpt_dir = os.path.dirname(ckpt_path)
    else:
      ckpt_dir = ckpt_path
  else:
    # during training/eval parallel jobs, use the current config.
    ckpt_dir = workdir
    train_config = config

  config = train_config

  if custom_outputs:
    result_dir = os.path.join(config.eval_from.result_dir,
                              config.eval_from.get('sub_dir', ''))
    ckpt_manager = TaskManager(
        result_dir,
        workdir,
        config.eval_from.output_pattern,
        lambda x: int(  # pylint: disable=g-long-lambda
            x.split('_')[config.eval_from.get('output_step_field', -1)]
        ),
    )
  else:
    ckpt_manager = TaskManager(ckpt_dir, workdir)
  if config.eval_from.get('step') is None:
    ckpt_list = ckpt_manager.unevaluated_checkpoints(
        timeout=config.eval_from.get('timeout', 3600 * 8))
  elif config.eval_from.step > 0:
    # Evaluates at a specified step
    ckpt_list = [
        c for c in ckpt_manager.unevaluated_checkpoints(0, return_all=True)
        if ckpt_manager.sort_key_fn(c) == config.eval_from.step
    ]
    assert ckpt_list, f'Checkpoint at step {config.eval_from.step} not found.'
  elif config.eval_from.step == -1:  # The last checkpoint.
    ckpt_list = [*ckpt_manager.unevaluated_checkpoints(0, return_all=True)][-1:]
    assert ckpt_list, 'No checkpoint found.'
  elif config.eval_from.step == 0:
    ckpt_list = [ckpt_dir]  # For unit test.
  else:
    raise ValueError(f'Invalid step to evaluate: {config.eval_from.step}')
  if final_num_repeats > 1:
    ckpt_list = _repeat_last(ckpt_list, final_num_repeats)
  return config, ckpt_manager, ckpt_list


class ReportEvaluationProgress(periodic_actions.ReportProgress):
  """Report evaluation progress."""

  def _apply(self, step: int, t: float):
    steps_per_sec = (step - self._previous_step) / (t - self._previous_time)
    if self._writer is not None:
      self._writer.write_scalars(step, {'steps_per_sec/eval': steps_per_sec})


def flatten_config(config: ml_collections.ConfigDict, prefix: str = ''):
  flattened = {}
  for k, v in config.items():
    if isinstance(v, list):
      flattened[prefix + k] = ', '.join(map(str, v))
    elif not isinstance(v, (dict, ml_collections.ConfigDict)):
      flattened[prefix + k] = v
    else:
      flattened.update(flatten_config(v, f'{prefix}{k}/'))
  return flattened


def get_ckpt_all_steps(checkpoints_dir: str) -> List[int]:
  """Returns a list of available step numbers in ascending order."""
  # Assumes the checkpoint has the format "checkpoint_\d+"
  glob_pattern = os.path.join(checkpoints_dir, 'checkpoint_*')
  checkpoint_paths = gfile.glob(glob_pattern)
  steps = []
  for each in checkpoint_paths:
    steps.append(int(each.split('_')[-1]))
  sorted(steps)
  return steps


def _strip_train_state_4eval(train_state: TrainState,
                             deny_list: Iterable[str]) -> TrainState:

  deny_list = list(deny_list) + ['global_step', 'rng', 'metadata']
  params = {}
  for k in train_state.__dict__.keys():
    if k not in deny_list:
      params[k] = None
  return train_state.replace(**params)


def strip_train_state_4eval(train_state: TrainState) -> TrainState:
  """Strips the train_state to bare minimum for evaluation."""
  train_state_type = type(train_state)
  if train_state_type == train_state_manager.VQGANTrainState:
    deny_list = ['g_params', 'g_model_state', 'ema_params']
  elif train_state_type == train_state_manager.ScenicTrainState:
    deny_list = ['params', 'model_state']
  else:
    raise ValueError('train_state is not supported.')
  return _strip_train_state_4eval(train_state, deny_list)


def load_metric_params(config: ml_collections.ConfigDict) -> Dict[str, Any]:
  """Loads the auxiliary parameters to compute the metrics."""
  metric_params = {}
  if config.eval.get('enable_inception_score', False):
    metric_params['inception_score'] = inception_score.load_params(
        config.eval.get('inception_score_checkpoint_filename'))
  if config.eval.get('enable_frechet_distance', False):
    metric_params['frechet_distance'] = frechet_distance.load_params(
        config.eval.get('frechet_distance_checkpoint_filename'))
  if config.eval.get('enable_lpips', False):
    lpips_model_dict = image_quality_metrics.load_lpips_models()
    metric_functions = dict(
        lpips_alexnet=functools.partial(
            image_quality_metrics.lpips_tf,
            lpips_model=lpips_model_dict['lpips_alexnet']),
        lpips_vgg=functools.partial(
            image_quality_metrics.lpips_tf,
            lpips_model=lpips_model_dict['lpips_vgg']))
    metric_params['lpips'] = metric_functions
  return metric_params


def eval_step_get_features(
    outputs: Dict[str, Any], *,
    metric_params: Dict[str, Any],
    model_suffix_list: Iterable[str] = ('',),
    config: ml_collections.ConfigDict) -> EvalFeatureDict:
  """Extract features to compute eval metrics.

  Args:
    outputs: Dict of original video, generated video and batch_mask.
    metric_params: Params for metric models.
    model_suffix_list: Tuple of model suffixes, '' for the default and '_ema'
      for the ema model.
    config: Configurations of the experiment.

  Returns:
    Extracted metric features.
  """
  features = {'batch_mask': outputs['batch_mask']}
  size_suffixes = ['']
  if config.eval.get('image_resize') is not None:
    suffix = f'_resize{config.eval.image_resize}'
    for key in [*outputs.keys()]:
      if 'video' in key:
        outputs[f'{key}{suffix}'] = metrics_lib.resize_bilinear(
            outputs[key], (config.eval.image_resize, config.eval.image_resize))
    size_suffixes.append(suffix)
    if config.eval.get('resized_only'):
      # Only evaluate resized samples, not original ones.
      size_suffixes.remove('')
  if config.eval.get('enable_inception_score', False):
    for s1, s2 in itertools.product(model_suffix_list, size_suffixes):
      suffix = s1 + s2
      features[f'inception_feature{suffix}'] = inception_score.run_model(
          metric_params['inception_score'],
          outputs[f'generated_video{suffix}'])
  if config.eval.get('enable_frechet_distance', False):
    for suffix in size_suffixes:
      if f'original_video{suffix}' not in outputs:
        continue
      features[f'frechet_feature_orig{suffix}'] = frechet_distance.run_model(
          metric_params['frechet_distance'], outputs[f'original_video{suffix}'],
          **config.eval.get('frechet_distance_args', {}))
    for s1, s2 in itertools.product(model_suffix_list, size_suffixes):
      suffix = s1 + s2
      if f'generated_video{suffix}' not in outputs:
        continue
      features[f'frechet_feature{suffix}'] = frechet_distance.run_model(
          metric_params['frechet_distance'],
          outputs[f'generated_video{suffix}'],
          **config.eval.get('frechet_distance_args', {}))
  if config.eval.get('enable_ssim_psnr', False):
    for s1, s2 in itertools.product(model_suffix_list, size_suffixes):
      suffix = s1 + s2
      cur_out_dict = image_quality_metrics.run_models(
          outputs[f'original_video{s2}'],
          outputs[f'generated_video{suffix}'],
          is_tf_function=False,
          metric_functions=None)
      features.update(
          {f'{k}{suffix}': v for k, v in cur_out_dict.items()})
      cur_out_dict = image_quality_metrics.run_models(
          outputs[f'generated_video{suffix}'][:, 1:],
          outputs[f'generated_video{suffix}'][:, :-1],
          is_tf_function=False,
          metric_functions=None)
      features.update(
          {f'interframe_{k}{suffix}': v for k, v in cur_out_dict.items()})
  if config.eval.get('enable_utilization', False):
    for suffix in model_suffix_list:
      if f'generated_tokens{suffix}' not in outputs:
        continue
      tokens = outputs[f'generated_tokens{suffix}']
      tokens = jax.tree_util.tree_map(
          lambda x: x.reshape(x.shape[0], -1), tokens)
      tokens = jnp.concatenate(jax.tree_leaves(tokens), axis=-1)
      features[f'tokens{suffix}'] = tokens

  features = jax.lax.all_gather(features, axis_name='device', tiled=True)
  return features  # total_batch_size, ...


def eval_step_get_features_cpu(
    outputs: Dict[str, Any], *,
    metric_params: Dict[str, Any],
    model_suffix_list: Iterable[str] = ('',),
    config: ml_collections.ConfigDict) -> EvalFeatureDictCPU:
  """Extract features on cpu to compute eval metrics.

  Args:
    outputs: Dict of original video, generated video and batch_mask.
    metric_params: Params for metric models.
    model_suffix_list: Tuple of model suffixes, '' for the default and '_ema'
      for the ema model.
    config: Configurations of the experiment.

  Returns:
    Extracted metric features. A dictionary mapping the feature name (e.g.
    fad_vggish_orig) to a dictionary of (str, np.array).
    Example:
    if config.eval.enable_fad_vggish == True
    features['fad_vggish_orig'] = {
        'vggish_embedding': [total_batch_size, 256]
        }
  """
  features = {}
  del metric_params, model_suffix_list


  if features:
    features = train_utils.all_gather_cpu(features)
  return features  # total_batch_size, ...


def _get_suffixes(features: EvalFeatureDictCPU, prefix: str) -> List[str]:
  return [k[len(prefix):] for k in features if k.startswith(prefix)]


def _mean_and_std(values: np.ndarray, num_repeats: int) -> Dict[str, float]:
  values = values.reshape(num_repeats, -1, *values.shape[1:])
  scores = values.mean(axis=1)
  return {'mean': scores.mean(), 'std': scores.std()}


def _max_in_repeats(values: np.ndarray, num_repeats: int) -> Dict[str, float]:
  assert values.shape[0] % num_repeats == 0
  values = values.reshape(num_repeats, -1, *values.shape[1:])
  scores = values.max(axis=0)
  return {'mean': scores.mean(), 'std': 0}


def compute_eval_metrics(
    eval_features: EvalFeatureDictCPU,
    config: ml_collections.ConfigDict,
    total_num_examples: int,
    num_repeats: int = 1,
    model_suffix_list: Iterable[str] = ('',),
) -> EvalMetricDict:
  """Calculate global metrics."""
  actual_size = jax.tree_util.tree_map(lambda x: x.shape[0], eval_features)
  actual_size['expected'] = total_num_examples
  num_totals = set(jax.tree_util.tree_leaves(actual_size))
  if len(num_totals) > 1:
    logging.warning('Inconsistent num examples: %s', actual_size)
  num_examples = total_num_examples // num_repeats
  eval_metrics = {'num_examples': num_examples, 'num_repeats': num_repeats}
  size_suffixes = ['']
  if config.eval.get('image_resize') is not None:
    size_suffixes.append(f'_resize{config.eval.image_resize}')
  if config.eval.get('enable_inception_score', False):
    suffixes = _get_suffixes(eval_features, 'inception_feature')
    for suffix in suffixes:
      score_dict = inception_score.inception_score_from_logits(
          eval_features[f'inception_feature{suffix}']['logits'],
          num_repeats)
      eval_metrics[f'inception_score{suffix}'] = score_dict['mean']
      eval_metrics[f'inception_score{suffix}-std'] = score_dict['std']
  if config.eval.get('enable_frechet_distance', False):
    size_suffixes = _get_suffixes(eval_features, 'frechet_feature_orig')
    for s1, s2 in itertools.product(model_suffix_list, size_suffixes):
      score_dict = frechet_distance.frechet_distance_from_logits(
          eval_features[f'frechet_feature{s1}{s2}']['logits_mean'],
          eval_features[f'frechet_feature_orig{s2}']['logits_mean'],
          num_repeats)
      eval_metrics[f'frechet_distance{s1}{s2}'] = score_dict['mean']
      eval_metrics[f'frechet_distance{s1}{s2}-std'] = score_dict['std']
  if config.eval.get('enable_ssim_psnr', False):
    reduce_fn = _max_in_repeats if config.eval.get(
        'ssim_psnr_max_in_repeats', False) else _mean_and_std
    size_suffixes = _get_suffixes(eval_features, 'ssim')
    for s1, s2 in itertools.product(['ssim', 'psnr'], size_suffixes):
      score_dict = reduce_fn(eval_features[f'{s1}{s2}'], num_repeats)
      eval_metrics[f'{s1}{s2}'] = score_dict['mean']
      eval_metrics[f'{s1}{s2}-std'] = score_dict['std']
  if config.eval.get('enable_lpips', False):
    suffixes = _get_suffixes(eval_features, 'lpips')
    for suffix in suffixes:
      score_dict = _mean_and_std(eval_features[f'lpips{suffix}'], num_repeats)
      eval_metrics[f'lpips{suffix}'] = score_dict['mean']
      eval_metrics[f'lpips{suffix}-std'] = score_dict['std']
  if config.eval.get('enable_fad_vggish', False):
    score_dict = frechet_distance.frechet_distance_from_logits(
        eval_features['fad_vggish_orig']['vggish_embedding'],
        eval_features['fad_vggish_generated']['vggish_embedding'],
        num_repeats,
    )
    eval_metrics['fad_vggish'] = score_dict['mean']
    eval_metrics['fad_vggish-std'] = score_dict['std']
  if config.eval.get('enable_fad_trill', False):
    score_dict = frechet_distance.frechet_distance_from_logits(
        eval_features['fad_trill_orig']['trill_embedding'],
        eval_features['fad_trill_generated']['trill_embedding'],
        num_repeats,
    )
    eval_metrics['fad_trill'] = score_dict['mean']
    eval_metrics['fad_trill-std'] = score_dict['std']
  if config.eval.get('enable_favd_vggsound', False):
    score_dict = frechet_distance.frechet_distance_from_logits(
        eval_features['favd_vggsound_orig'],
        eval_features['favd_vggsound_generated'],
        num_repeats,
    )
    eval_metrics['favd_vggsound'] = score_dict['mean']
    eval_metrics['favd_vggsound-std'] = score_dict['std']
  if config.eval.get('enable_utilization', False):
    suffixes = _get_suffixes(eval_features, 'tokens')
    for suffix in suffixes:
      count = np.zeros(config.vqvae.codebook_size, dtype=np.int32)
      np.add.at(count, eval_features[f'tokens{suffix}'].reshape(-1), 1)
      utilization = (count > 0).mean()
      p10, p90 = np.percentile(count / count.mean(), [10, 90])
      eval_metrics[f'token_utilization{suffix}'] = utilization
      eval_metrics[f'token_utilization_p10{suffix}'] = p10
      eval_metrics[f'token_utilization_p90{suffix}'] = p90
  return eval_metrics


def load_label_names(
    config: ml_collections.ConfigDict) -> Optional[Dict[int, str]]:
  """Load label name mapping for the current dataset."""
  dataset_name = config.dataset_configs.get('dataset_name', config.dataset_name)
  if config.dataset_configs.get('label_dir') is None:
    return
  label_path = f'{config.dataset_configs.label_dir}/{dataset_name}_labels.json'
  if not gfile.exists(label_path):
    return
  with gfile.GFile(label_path) as f:
    label_names = json.load(f)
  label_names = {int(k): v for k, v in label_names.items()}
  return label_names


def get_result_dir(config: ml_collections.ConfigDict) -> str:
  """Create result directory."""
  result_dir = config.eval.results_dir
  if not gfile.exists(result_dir):
    gfile.makedirs(result_dir)
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY,
      artifact=result_dir,
      description='Result directory')
  return result_dir




def get_local_batch(batch: Batch) -> Batch:
  """Slice local batch from an all_gathered batch."""
  global_bs = jax.tree_util.tree_leaves(batch)[0].shape[0]
  local_bs = global_bs // jax.process_count()
  proc_i = jax.process_index()
  return jax.tree_util.tree_map(
      lambda x: x[local_bs * proc_i:local_bs * (proc_i + 1)], batch)


def apply_batch_mask(batch: Batch) -> Batch:
  """Only keep valid samples in a flattened batch."""
  if 'batch_mask' in batch:
    mask = batch.pop('batch_mask') > 0
    batch = jax.tree_util.tree_map(lambda x: x[mask], batch)
  return batch


def write_batch_samples(batch: Batch, step: int,
                        executor: concurrent.futures.Executor,
                        result_dir: str, prefix: str,
                        label_names: Optional[Dict[int, str]],
                        config: ml_collections.ConfigDict):
  """Write a batch of samples to disk."""
  enable_condition = config.eval.get('results_with_condition', True)
  enable_input = config.eval.get('results_with_input', False)
  condition_alpha = config.eval.get('results_condition_alpha',
                                    0 if 'frame' not in prefix else 1)
  batch = apply_batch_mask(batch)
  if 'generated_video_ema' in batch:  # For VQ models
    videos = batch['generated_video_ema']
    suffix = '_ema'
    if enable_condition:
      videos = np.concatenate((videos, batch['original_video']), axis=2)
  else:  # For GVT models
    videos = batch['generated_video']
    suffix = ''
    if enable_input:
      videos = np.concatenate((videos, batch['original_video']), axis=2)
    if 'condition_video' in batch and enable_condition:
      cond_videos = batch['condition_video']
      if 'frame_interpolation' in prefix:
        # TODO(Lijun-Yu): refactor this visualization hack.
        l = cond_videos.shape[1] // 2
        cond_videos[:, 1:l] = cond_videos[:, :1]
        cond_videos[:, l:-1] = cond_videos[:, -1:]
      cond_videos = np.where(batch['condition_mask'], batch['condition_video'],
                             condition_alpha * cond_videos)
      videos = np.concatenate((videos, cond_videos), axis=2)
  label_ids = batch.get('label')
  if label_ids is not None and label_ids.shape[1] > 1:
    # Take the first one hot label
    label_ids = label_ids.argmax(axis=1, keepdims=True).astype(np.int32)
  idx_offset = step * config.eval_batch_size + jax.process_index(
  ) * config.eval_batch_size // jax.process_count()
  if 'frame_rate' in config.dataset_configs:
    fps = config.dataset_configs.frame_rate / config.dataset_configs.stride
  else:
    fps = 8
  for idx, video in enumerate(videos):
    label_str = ''
    if label_names is not None and label_ids is not None:
      label_id = int(label_ids[idx, 0])
      label_str = f'{label_id:04d}{label_names[label_id]}_'
    if config.eval.get('enable_ssim_psnr', False):
      label_str += f'ifssim={batch[f"interframe_ssim{suffix}"][idx]:.02f}_'
      # label_str += f'ssim={batch[f"ssim{suffix}"][idx]:.02f}_'
      # label_str += f'psnr={batch[f"psnr{suffix}"][idx]:02.0f}_'
    if config.eval.get('enable_lpips', False):
      # label_str += f'lpalex={batch[f"lpips_alexnet"][idx]:.02f}_'
      label_str += f'lpvgg={batch["lpips_vgg"][idx]:.02f}_'
    filename = f'{prefix}_{label_str}{(idx + idx_offset):09d}.mp4'
    path = os.path.join(result_dir, filename)
    executor.submit(mediapy.write_video, path, video, fps=fps)


def write_video_with_audio(
    output_path: str,
    video_frames: np.ndarray,
    audio_samples: Optional[np.ndarray] = None,
    *,
    fps: float,
    audio_sample_rate: int = 16000,
):
  """Write video potentially with audio."""
  if audio_samples is None:
    mediapy.write_video(output_path, video_frames, fps=fps)
    return
  with tempfile.TemporaryDirectory() as tmp_dir:
    audio_path = os.path.join(tmp_dir, 'audio.wav')
    video_path = os.path.join(tmp_dir, 'video.mp4')
    temp_output_path = os.path.join(tmp_dir, 'output.mp4')
    wavfile.write(audio_path, audio_sample_rate, audio_samples)
    mediapy.write_video(video_path, video_frames, fps=fps)

    config = subprocess_pb2.SubprocessConfig()
    params = ffmpeg_pb2.FfmpegParams()
    params.input.add().file_name = video_path
    params.input.add().file_name = audio_path
    ffmpeg_output = params.output.add()
    ffmpeg_output.container.format = ffmpeg_pb2.FfmpegParams.FORMAT_ID_MP4
    ffmpeg_output.audio.add()
    ffmpeg_output.audio[0].sample_rate = audio_sample_rate
    ffmpeg_output.audio[0].bitrate = audio_sample_rate
    ffmpeg_output.video.add()
    ffmpeg_output.file_name = temp_output_path
    set_ffmpeg_binary_flags.SetFfmpegBinaryFlags()
    executor = ffmpeg.Ffmpeg()
    success = False
    if executor.Init(config, params):
      success = executor.Run()
    if success:
      gfile.copy(temp_output_path, output_path, overwrite=True)
    else:
      raise ValueError(f'Failed to save video {output_path}.')


def write_batch_samples_with_audio(
    batch: Batch,
    step: int,
    executor: concurrent.futures.Executor,
    result_dir: str,
    prefix: str,
    label_names: Optional[Dict[int, str]],
    config: ml_collections.ConfigDict,
):
  """Write a batch of samples to disk."""
  enable_condition = config.eval.get('results_with_condition', True)
  enable_target = config.eval.get('results_with_target', True)
  batch = apply_batch_mask(batch)
  label_ids = batch.get('label')
  if label_ids is not None and label_ids.shape[1] > 1:
    # Take the first one hot label
    label_ids = label_ids.argmax(axis=1, keepdims=True).astype(np.int32)
  idx_offset = step * config.eval_batch_size + jax.process_index(
  ) * config.eval_batch_size // jax.process_count()
  if 'frame_rate' in config.dataset_configs:
    fps = config.dataset_configs.frame_rate / config.dataset_configs.stride
  else:
    fps = 8
  audio_sample_rate = config.dataset_configs.audio_sample_rate
  batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
  default_audio = batch.get('condition_audio', [None] * batch_size)
  dummy_video = np.zeros(
      (batch_size, config.dataset_configs.num_frames, 16, 16, 3), dtype=np.uint8
  )
  default_video = batch.get('condition_video', dummy_video)
  batch = batch.copy()  # shallow copy to fill in default values
  for key in ['condition', 'generated', 'target']:
    batch.setdefault(f'{key}_audio', default_audio)
    batch.setdefault(f'{key}_video', default_video)
  write_fn = functools.partial(executor.submit, write_video_with_audio, fps=fps,
                               audio_sample_rate=audio_sample_rate)
  for idx in range(batch_size):
    label_str = ''
    if label_names is not None and label_ids is not None:
      label_id = int(label_ids[idx, 0])
      label_str = f'{label_id:04d}{label_names[label_id]}_'
    filename = f'{prefix}_{label_str}{(idx + idx_offset):09d}'
    path = os.path.join(result_dir, filename)
    write_fn(f'{path}_gen.mp4', batch['generated_video'][idx],
             batch['generated_audio'][idx])
    if enable_condition:
      write_fn(f'{path}_cond.mp4', batch['condition_video'][idx],
               batch['condition_audio'][idx])
    if enable_target:
      write_fn(f'{path}_target.mp4', batch['target_video'][idx],
               batch['target_audio'][idx])
    if 'generated_video_sr' in batch:
      write_fn(
          f'{path}_gen_sr.mp4',
          batch['generated_video_sr'][idx],
          batch['generated_audio'][idx],
      )
