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

"""Utility functions for Colab."""

import base64
import functools
import os
import re
from typing import Any, Dict, List, Tuple, Union

from absl import logging
import flax.linen as nn
from IPython import display
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy
import ml_collections
import numpy as np
from scenic.train_lib import lr_schedules
import tensorflow as tf
from tensorflow.io import gfile
from videogvt.interfaces import mm_vq
from videogvt.train_lib import train_utils


def tokenize_long_video(
    long_video: np.ndarray,
    tokenizer_dict: Dict[str, nn.Module],
    temporal_window: int = 16,
    overlapping_frames: int = 0,
    batch_size: int = 8) -> Dict[str, Union[np.ndarray, jnp.ndarray]]:
  """Tokenize and detokenize a long video using the tokenizer_dict.

  This is supposed to be called only once and hence not optimized for
  efficiency.

  Args:
    long_video: A single long video of shape [frames, h, w, c]
    tokenizer_dict: A dictionary of tokenizer and detokenizer
    temporal_window: Temporal window to chunk the input video for 3D Conv.
    overlapping_frames: How many frames are overlapping between the windows.
    batch_size: Batch size to run the tokenizer. Must be divisiable by
      jax.device_count().

  Returns:
    A dictionary of original video, detokenized/reconstructed video,
      and corresponding visual codes. It will also return pmaped tokenizer,
      and detokenizer if return_pmapped_func=True.
  """
  assert long_video.ndim == 4, 'Video format should be [#frames, h, w, c].'
  assert np.min(long_video) >= 0 and np.max(
      long_video) < 1 + 1e-1, 'Video value range is wrong.'
  assert batch_size % jax.device_count() == 0, 'Not divisiable by device_count'

  if 'tokenizer_pmapped' not in tokenizer_dict:
    tokenizer_pmapped = jax.pmap(
        functools.partial(mm_vq.tokenize, tokenizer_dict=tokenizer_dict),
        axis_name='device')
    tokenizer_dict['tokenizer_pmapped'] = tokenizer_pmapped

  if 'detokenizer_pmapped' not in tokenizer_dict:
    detokenizer_pmapped = jax.pmap(
        functools.partial(mm_vq.detokenize, tokenizer_dict=tokenizer_dict),
        axis_name='device')
    tokenizer_dict['detokenizer_pmapped'] = detokenizer_pmapped

  eval_iter = iter(
      train_utils.LongVideoIterable(long_video, batch_size=batch_size,
                                    temporal_window=temporal_window,
                                    overlapping_frames=overlapping_frames))
  all_tokens = []
  # Uses np not jnp as much as possible to save TPU memory.
  local_batch_size = batch_size // jax.device_count()
  for train_batch in eval_iter:
    train_batch = jax.tree_util.tree_map(lambda x: np.reshape(  # pylint: disable=g-long-lambda
        x, [-1, local_batch_size, *x.shape[1:]]), train_batch)
    tokens = jax.device_get(tokenizer_dict['tokenizer_pmapped'](train_batch))

    logging.log_first_n(logging.INFO, 'Input video shape %s.', 1,
                        train_batch['inputs'].shape)
    logging.log_first_n(logging.INFO, 'Tokens shape %s.', 1, tokens.shape)
    all_tokens.append(tokens.reshape(-1, *tokens.shape[2:]))
  all_tokens = np.concatenate(all_tokens)

  all_reconstructed = reconstruct_from_tokens(all_tokens, tokenizer_dict,
                                              overlapping_frames)
  all_reconstructed = all_reconstructed.reshape(-1, *long_video.shape[1:])
  video_len = min(all_reconstructed.shape[0], long_video.shape[0])
  result = dict(
      input=long_video[0:video_len],
      reconstructed=all_reconstructed[0:video_len],
      tokens=all_tokens)
  return result


def pad(x: np.ndarray, num_devices: int, constant: float = 0.0):
  """Pads x of shape [bs, t, h, w] into [device_bs, #device, t, h, w]."""
  assert (
      x.ndim == 4 or x.ndim == 5
  ), 'Not in [bs, t, h, w] or [bs, t, h, w, rvq]'
  num_batches = int(np.ceil(x.shape[0] * 1.0 / num_devices))
  num_to_pad = num_batches * num_devices - x.shape[0]
  pad_values = np.ones_like(x[0]) * constant
  if num_to_pad > 0:
    logging.log_first_n(logging.WARNING, 'Padding %d frames with shape %s.', 20,
                        num_to_pad, pad_values.shape)
    padded_x = np.concatenate((x, [pad_values] * num_to_pad))
  else:
    padded_x = x
  padded_x = padded_x.reshape([-1, num_devices, *x.shape[1:]])
  return padded_x.astype(x.dtype)


def reconstruct_from_tokens(segmented_tokens: np.ndarray,
                            tokenizer_dict: Dict[str, nn.Module],
                            overlapping_frames: int):
  """Reconstructs a long video using the segmented tokens.

  Args:
    segmented_tokens: Array of shape [bs, l_t, l_h, l_w] of overlapping tokens.
    tokenizer_dict: A dictionary of tokenizer and detokenizer
    overlapping_frames: How many frames are overlapping between the windows.

  Returns:
    all_reconstructed: RGB video of shape [bs, t, h, w, 3].
  """
  assert (
      segmented_tokens.ndim == 4 or segmented_tokens.ndim == 5
  ), 'Not in [bs, l_t, l_h, l_w] or [bs, l_t, l_h, l_w, rvq]'
  padded_segmented_tokens = pad(segmented_tokens, jax.device_count())

  if 'detokenizer_pmapped' not in tokenizer_dict:
    detokenizer_pmapped = jax.pmap(
        functools.partial(mm_vq.detokenize, tokenizer_dict=tokenizer_dict),
        axis_name='device')
    tokenizer_dict['detokenizer_pmapped'] = detokenizer_pmapped
  all_reconstructed = []
  for i in range(padded_segmented_tokens.shape[0]):
    tokens = padded_segmented_tokens[i]
    tokens = tokens[:, None, ...]  # in [#device, 1, l_t, l_h, l_w]
    reconstructed_videos = tokenizer_dict['detokenizer_pmapped'](tokens)
    reconstructed_videos = jax.device_get(reconstructed_videos)
    reconstructed_videos = np.array(reconstructed_videos, copy=False)  # asarray
    logging.log_first_n(logging.INFO, 'Reconstructed video shape %s.', 1,
                        reconstructed_videos.shape)
    all_reconstructed.append(reconstructed_videos)
  all_reconstructed = np.concatenate(all_reconstructed)
  all_reconstructed = all_reconstructed.reshape(
      [-1, *all_reconstructed.shape[2:]])  # collaps the device dim
  all_reconstructed = all_reconstructed[0:segmented_tokens.shape[0]]

  if overlapping_frames > 0:
    all_reconstructed = train_utils.pool_overlapping_segments(
        all_reconstructed, overlapping_frames)
  return all_reconstructed


def tokenize_long_video_double_fusion(
    long_video: np.ndarray,
    tokenizer_dict: Dict[str, nn.Module],
    temporal_window: int = 16,
    overlapping_frames: int = 8,
    batch_size: int = 8,
    mode: str = 'early') -> Dict[str, Union[np.ndarray, jnp.ndarray]]:
  """Tokenize a long video using fusion of pixel and embeddings."""
  # Experimental. Needs further research to find the pooling layer.
  assert long_video.ndim == 4, 'Video format should be [#frames, h, w, c].'
  assert np.min(long_video) >= 0 and np.max(
      long_video) < 1 + 1e-1, 'Video value range is wrong.'
  assert batch_size % jax.device_count() == 0, 'Not divisiable by device_count'

  if 'tokenizer_pmapped' not in tokenizer_dict:
    tokenizer_pmapped = jax.pmap(
        functools.partial(mm_vq.tokenize, tokenizer_dict=tokenizer_dict),
        axis_name='device')
    tokenizer_dict['tokenizer_pmapped'] = tokenizer_pmapped

  if 'detokenizer1_pmapped' not in tokenizer_dict:
    tokenizer_dict['detokenizer1_pmapped'] = jax.pmap(
        functools.partial(tokenizer_dict['detokenizer1']),
        axis_name='device')

  if 'detokenizer2_pmapped' not in tokenizer_dict:
    tokenizer_dict['detokenizer2_pmapped'] = jax.pmap(
        functools.partial(tokenizer_dict['detokenizer2']),
        axis_name='device')

  eval_iter = iter(
      train_utils.LongVideoIterable(long_video, batch_size=batch_size,
                                    temporal_window=temporal_window,
                                    overlapping_frames=overlapping_frames))
  all_tokens = []
  all_embeddings = []
  local_batch_size = batch_size // jax.device_count()
  for train_batch in eval_iter:
    train_batch = jax.tree_util.tree_map(lambda x: np.reshape(  # pylint: disable=g-long-lambda
        x, [-1, local_batch_size, *x.shape[1:]]), train_batch)
    tokens = jax.device_get(tokenizer_dict['tokenizer_pmapped'](train_batch))
    embeddings = tokenizer_dict['detokenizer1_pmapped'](tokens)
    embeddings = jax.device_get(embeddings)
    embeddings = np.array(embeddings, copy=False)  # asarray
    all_tokens.append(tokens.reshape(-1, *tokens.shape[3:]))
    all_embeddings.append(embeddings)

  all_tokens = np.concatenate(all_tokens)
  all_embeddings = np.concatenate(all_embeddings)
  if overlapping_frames > 0:
    x = all_embeddings
    x = x.reshape([-1, *x.shape[2:]])  # collaps the device dim
    all_embeddings = train_utils.pool_overlapping_segments(
        x, overlapping_frames, alphas=None)

  # decode second stage
  all_embeddings = pad(all_embeddings, jax.device_count())
  all_reconstructed = []
  for i in range(all_embeddings.shape[0]):
    all_reconstructed.append(tokenizer_dict['detokenizer2_pmapped'](
        all_embeddings[i]))

  all_reconstructed = np.array(all_reconstructed)

  if overlapping_frames and mode == 'double':
    x = all_reconstructed
    x = x.reshape([-1, *x.shape[2:]])  # collaps the device dim
    all_reconstructed = train_utils.pool_overlapping_segments(
        x, overlapping_frames)

  all_reconstructed = all_reconstructed.reshape(-1, *long_video.shape[1:])

  result = dict(
      input=long_video[0:all_reconstructed.shape[0]],
      reconstructed=all_reconstructed,
      tokens=all_tokens)

  return result


def show_mm_table(input_dict: Dict[str, Any],
                  fps: int = 8,
                  sound_sr: int = 16000,
                  max_num_show=25):
  """Shows the generation result in a multimodal table."""
  generated_video = input_dict['generated_video']
  generated_video_sxs = input_dict.get('generated_video_sxs')
  condition_video = input_dict.get('condition_video')
  truth_video = input_dict.get('original_video')
  yt_ids = input_dict.get('yt_ids')
  audio = input_dict.get('audio')

  def _check_shape():
    num_columns = 1 + 1
    assert generated_video.ndim == 5, 'Video format should be [bs, t, h, w, c].'
    if condition_video is not None:
      assert generated_video.shape[0] == condition_video.shape[0]
      num_columns += 1
    if generated_video_sxs is not None:
      assert generated_video.shape == generated_video_sxs.shape
      num_columns += 1
    if truth_video is not None:
      assert generated_video.shape == truth_video.shape
      num_columns += 1
    if audio is not None:
      assert generated_video.shape[0] == audio.shape[0]
      num_columns += 1
    if yt_ids:
      assert len(yt_ids) == generated_video.shape[0]
      num_columns += 1
    return num_columns

  num_samples = min(max_num_show, generated_video.shape[0])
  num_columns = _check_shape()
  # Colabtools are not supported outside of a notebook context, so we
  # lazily import from it to avoid import errors when using this file
  # as a library.
  from colabtools import table  # pylint: disable=g-import-not-at-top
  mm_table = table.Table(rows=num_samples + 1, columns=num_columns)
  for cur_row in range(num_samples):
    cur_col = 0
    mm_table.SetActive(cur_row, cur_col)
    print(f'example {cur_row}')
    if condition_video is not None:
      cur_col += 1
      mm_table.SetActive(cur_row, cur_col)
      print('condition')
      mediapy.show_video(condition_video[cur_row], codec='gif', fps=fps)
    # generated video
    cur_col += 1
    mm_table.SetActive(cur_row, cur_col)
    print('generation')
    mediapy.show_video(generated_video[cur_row], codec='gif', fps=fps)
    if generated_video_sxs is not None:
      cur_col += 1
      mm_table.SetActive(cur_row, cur_col)
      print('gen_sxs')
      mediapy.show_video(generated_video_sxs[cur_row], codec='gif', fps=fps)
    if truth_video is not None:
      cur_col += 1
      mm_table.SetActive(cur_row, cur_col)
      print('truth')
      mediapy.show_video(truth_video[cur_row], codec='gif', fps=fps)
    if audio is not None:
      cur_col += 1
      mm_table.SetActive(cur_row, cur_col)
      from colabtools import sound  # pylint: disable=g-import-not-at-top
      sound.Play(np.squeeze(audio[cur_row]), sound_sr=sound_sr)
    if yt_ids:
      cur_col += 1
      mm_table.SetActive(cur_row, cur_col)
      yt_id = yt_ids[cur_row]
      from colabtools import publish  # pylint: disable=g-import-not-at-top
      publish.html(f'<a href="https://www.youtube.com/embed/{yt_id}?start=0' +
                   '&end=2" target="_blank">YT</a>')


def read_from_tf_event(input_event_file_pattern: str,
                       metric2id_dict: Dict[str, int],
                       biggest: bool = True,
                       missing_value: float = 0.0) -> Dict[int, np.ndarray]:
  """Reads metrics from the TF event files in input_event_file_pattern.

  Args:
    input_event_file_pattern: Pattern of the TensorFlow event files for a single
      experiment.
    metric2id_dict: Dict of the metric names to their index
    biggest: Whether to read only from the biggest event file. In some cases,
      biggest file is also the latest. If False, read from all event files.
    missing_value: Default values for missing fields.

  Returns:
    2D numpy array containing the step size and value extracted
      from the tag.
  """
  files = gfile.glob(input_event_file_pattern)
  ret_dict = {}

  # Sorts the file by file size. Bigger files are more recent.
  # The assumption may not hold in many jobs.
  # It usually holds for the eval_only or universal_evaluator jobs (hyper-tune).
  sizes = [gfile.stat(t).length for t in files]
  file_tuple = list(zip(files, sizes))
  file_tuple.sort(key=lambda tup: -tup[1])

  nan_array = np.empty([len(metric2id_dict)])
  nan_array[:] = np.nan

  if not file_tuple:
    return {0: nan_array}

  if biggest:
    file_tuple = [file_tuple[0]]

  for event_file in file_tuple:
    try:
      for summary in tf.compat.v1.train.summary_iterator(event_file[0]):
        for v in summary.summary.value:
          if v.tag in metric2id_dict:
            cur_ret_values = ret_dict.get(
                summary.step,
                np.ones([len(metric2id_dict)]) * missing_value)
            cur_ret_values[metric2id_dict[v.tag]] = tf.make_ndarray(v.tensor)
            ret_dict[summary.step] = cur_ret_values
    except tf.errors.NotFoundError:
      continue
  if not ret_dict:
    ret_dict = {0: nan_array}
  return ret_dict


def auto_find_exps(all_exp_pattern: str,
                   re_pattern: str = '(.+)$') -> Tuple[List[str], List[str]]:
  """Finds all validate experiment names and their directories.

  Args:
    all_exp_pattern: Pattern to find all the experiments.
    re_pattern: Regular expression of the experiment dir to find. Use '()' to
      capture the  experiment ID. E.g.
      all_exp_pattern = '/path/to/exp_name/43951735/*'
      re_pattern = '.*/(10.*)$'
      It will find all the wid under 43951735 starting with 10 such as
      10, 100, 101, ... and use the (10.*) as the experiment name.

  Returns:
    exp_names: 1D list of experiment names.
    exp_dirs: 1D list of experiment absolute paths.
  """
  files = gfile.glob(all_exp_pattern)
  files.sort()
  exp_names, exp_dirs = [], []
  for f in files:
    cur_match = re.match(re_pattern, f)
    if cur_match:
      exp_names.append(cur_match.group(1))
      exp_dirs.append(os.path.join(cur_match.group(0), 'events*'))

  exp_names, exp_dirs = zip(*sorted(zip(exp_names, exp_dirs)))
  for i in range(len(exp_dirs)):
    assert exp_names[i] in exp_dirs[
        i], f'Unmatched {exp_names[i]} and {exp_dirs[i]}'
  return exp_names, exp_dirs


def get_outpainting_composite_mask(video_height: int,
                                   video_width: int,
                                   uncrop_t: int = 0,
                                   uncrop_b: int = 0,
                                   uncrop_l: int = 0,
                                   uncrop_r: int = 0,
                                   blending_pixels: int = 8):
  """Get outpainting composite mask with linear alpha blending.

  Use in this way.
  output = composit_mask * org_video + (1.0 - composit_mask) * gen_video
  for videos of shape (frames, height, width, 3)

  Args:
    video_height: Height of the video
    video_width: Width of the video
    uncrop_t: Number of uncrop pixels from the top
    uncrop_b: Number of uncrop pixels from the bottom
    uncrop_l: Number of uncrop pixels from the left
    uncrop_r: Number of uncrop pixels from the right
    blending_pixels: Number of pixels to do alpha blending

  Returns:
    Composite mask to use to merge original and new video.
  """
  composit_mask = np.zeros((1, video_height, video_width, 3))
  composit_mask[:, uncrop_t:video_height - uncrop_b,
                uncrop_l:video_width - uncrop_r] = 1
  for i in range(blending_pixels):
    if uncrop_t != 0:
      composit_mask[:, uncrop_t + i,
                    uncrop_l:video_width - uncrop_r] = i / blending_pixels
    if uncrop_b != 0:
      composit_mask[:, video_height - uncrop_b - i - 1,
                    uncrop_l:video_width - uncrop_r] = i / blending_pixels
    if uncrop_l != 0:
      composit_mask[:, uncrop_t:video_height - uncrop_b,
                    uncrop_l + i] = i / blending_pixels
    if uncrop_r != 0:
      composit_mask[:, uncrop_t:video_height - uncrop_b,
                    video_width - uncrop_r - i - 1] = i / blending_pixels
  return composit_mask


def get_model_steps(x: np.ndarray):
  assert len(x) >= 1
  re_pattern = '.+\\((\\d+)\\)$'
  if not re.match(re_pattern, x[0][0]):
    return None
  all_steps = []
  for i in range(x.shape[0]):
    all_steps.append(int(re.match(re_pattern, x[i][0]).group(1)))
  return np.array(all_steps)


def report_best_and_last(x: np.ndarray, lower_the_better: bool = True):
  """Finds the best and the last result of an experiment."""
  #  In format [exp_str(step), metric1, metric2, ...]
  assert lower_the_better
  all_training_steps = get_model_steps(x)
  if all_training_steps is not None:
    x = x[np.argsort(all_training_steps)]
  ds = np.copy(x[:, 1:])
  last_results = []
  for j in range(ds.shape[1]):
    last_nonzero_rowid = np.nonzero(ds[:, j])[0]
    if len(last_nonzero_rowid) != 0:  # pylint: disable=g-explicit-length-test
      last_results.append(ds[np.max(last_nonzero_rowid), j])
    else:
      last_results.append(np.inf)
  ds[np.where(ds == 0)] = np.inf
  best_results = []
  for j in range(ds.shape[1]):
    best_results.append(ds[np.nanargmin(ds[:, j]), j])
  return dict(last=last_results, best=best_results)


def plot_learning_rate(config: ml_collections.ConfigDict()):
  """Plots the learning rate in the config.

  Example:
    config1 = vqgan3d_videocc_ft_config.get_config('L')
    config2 = vqgan3d_ucf101_config.get_config()

    colab_utils.plot_learning_rate(config1)

    colab_utils.plot_learning_rate(config2)
    lr_configs = config_utils.get_lr_rsqrt_configs(
        0,
        config1,
        timescale=1_000,
        warmup_steps=0,
        cooldown_steps=15_000,
    )
    colab_utils.plot_learning_rate(lr_configs)

  Args:
    config: a config containing the lr_configs or config is the lr_configs.
  """
  if 'lr_configs' not in config:
    # config is lr_configs
    assert 'factors' in config, 'config does not have a lr_configs field.'
    config = ml_collections.ConfigDict(dict(lr_configs=config))
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  if 'total_steps' in config.lr_configs:
    total_steps = config.lr_configs.total_steps
  elif 'steps_per_cycle' in config.lr_configs:
    total_steps = config.lr_configs.steps_per_cycle    # consine lr
  else:
    raise NotImplementedError('missing total_steps in the learning rate.')

  xs = np.arange(1, total_steps, 100)
  plt.plot(xs, learning_rate_fn(xs))


def colab_show_videos_mp4(
    video_paths: Tuple[str, ...],
    video_width: int = 128,
    autoplay: bool = False,
    muted: bool = True,
):
  """Shows the video grid in the Colab.

  Args:
    video_paths: a list of video paths to show.
    video_width: the width of the videos.
    autoplay: auto play the video in a loop. Not use it for videos with sound.
    muted: Whether to mute the video by default

  Returns:
    a HTML component to show in the colab.
  """
  results = []
  if isinstance(video_paths, str):
    video_paths = [video_paths]

  video_flags = 'controls '
  if autoplay:
    video_flags += 'autoplay loop '
  if muted:
    video_flags += 'muted '

  for path in video_paths:
    if not path.endswith('mp4'):
      continue
    with open(path, 'r+b') as f:
      video_file = f.read()
    video_url = (
        f'data:video/mp4;base64,{base64.b64encode(video_file).decode()}'
    )
    results.append(f"""<video width={video_width} {video_flags}>
                   <source src="{video_url}"></video>""")
  return display.HTML('\n'.join(results))
