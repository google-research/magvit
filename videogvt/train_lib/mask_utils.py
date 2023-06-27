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

"""Lib for token mask strategies."""
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any
TokenDict = dict[str, jnp.ndarray]
MaskFn = Callable[..., TokenDict]
Limit = tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]
MASK_TOKEN = -1
NEVER_MASK = jnp.inf


def batch_flatten(x: jnp.ndarray) -> jnp.ndarray:
  return jnp.reshape(x, (x.shape[0], -1))


def batch_flatten_np(x: np.ndarray) -> np.ndarray:
  return np.reshape(x, (x.shape[0], -1))


def concat_tokens(*tokens_list: PyTree) -> PyTree:
  tokens_list = jax.tree_util.tree_map(batch_flatten, tokens_list)
  return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=-1),
                                *tokens_list)


def merge_limits(limit_1: Limit, limit_2: Limit) -> Limit:
  """Get the intersection of two limits."""
  min_1, max_1 = limit_1
  min_2, max_2 = limit_2
  if min_1 is not None and min_2 is not None:
    min_new = jnp.maximum(min_1, min_2)
  else:
    min_new = min_1 if min_1 is not None else min_2
  if max_1 is not None and max_2 is not None:
    max_new = jnp.minimum(max_1, max_2)
  else:
    max_new = max_1 if max_1 is not None else max_2
  limit_new = (min_new, max_new)
  return limit_new


def normalize_score(score: jnp.ndarray) -> jnp.ndarray:
  """Normalize score to range [0, 1] for each sample, input shape [B T H W]."""
  assert score.ndim == 4
  sample_min = score.min(axis=(1, 2, 3), keepdims=True)
  sample_max = score.max(axis=(1, 2, 3), keepdims=True)
  normalized_score = (score - sample_min) / (sample_max - sample_min + 1e-6)
  return normalized_score


def no_mask(batch: TokenDict,
            rng: Optional[jnp.ndarray] = None,
            mask_ratio: Optional[jnp.ndarray] = None) -> TokenDict:
  """Mask no tokens."""
  del rng, mask_ratio
  return {
      'masked_inputs': batch['inputs'],
      'uncond_masked_inputs': batch.get('uncond_inputs', batch['inputs']),
      'segment_ids': jnp.zeros_like(batch['inputs']),
      'targets': batch['inputs'],
      'weights': jnp.zeros(batch['inputs'].shape, jnp.float32),
  }


def all_mask(batch: TokenDict,
             rng: Optional[jnp.ndarray] = None,
             mask_ratio: Optional[jnp.ndarray] = None,
             *,
             weight: float = 1.) -> TokenDict:
  """Mask all tokens, except for the condition tokens."""
  del rng, mask_ratio
  if batch.get('cond_inputs') is not None:
    masked_inputs = jnp.where(batch['cond_mask'], batch['cond_inputs'],
                              MASK_TOKEN)
  else:
    masked_inputs = jnp.full_like(batch['inputs'], MASK_TOKEN)
  uncond_masked_inputs = jnp.full_like(batch['inputs'], MASK_TOKEN)
  return {
      'masked_inputs': masked_inputs,
      'uncond_masked_inputs': uncond_masked_inputs,
      'segment_ids': jnp.zeros_like(batch['inputs']),
      'targets': batch['inputs'],
      'weights': jnp.full_like(masked_inputs, weight, jnp.float32),
  }


def random_mask(
    batch: TokenDict,
    rng: Optional[jnp.ndarray],
    mask_ratio: jnp.ndarray,
    *,
    condition_mode: Optional[str] = None,
    mask_score: Optional[jnp.ndarray] = None,
    num_masked_limits: Limit = (None, None),
    condition_segment_id: int = 1,
    weight_mode: Optional[str] = None,
) -> TokenDict:
  """Random mask on 3D input tokens.

  Args:
    batch: Dictionary where batch['inputs'].shape=[batch, T, H, W] is the full
      tokens, and optionally batch['cond_inputs'] and batch['cond_mask'] in a
      broadcastable shape to batch['inputs'] are the condition tokens and masks.
    rng: PRNGKey.
    mask_ratio: Proportion of maskable tokens to mask out.
    condition_mode: Mode to handle condition tokens:
      None - No condition tokens.
      prefix - Prefix condition tokens, not in current inputs.
      cond->input - Use condition tokens to replace masked positions in
        condition region, requires batch['cond_inputs'] and batch['cond_mask'].
      input->input - Condition region is unmaskable, requires
        batch['cond_mask'].
      cond->cond - Condition region is unmaskable and is replaced with condition
        tokens, requires batch['cond_inputs'] and batch['cond_mask'].
    mask_score: Optionally provided mask score, otherwise mask score is
      uniformly sampled. Tokens with mask scores lower than a threshold are
      masked, where the threshold is determined according to mask_ratio and
      num_masked_limits.
    num_masked_limits: Optional limits on the number of masked tokens.
    condition_segment_id: Segment id for condition tokens.
    weight_mode: Mode to apply loss weights:
      None or False or mask+refine+recons - All maskable tokens.
      True or mask+refine - Only [MASK] tokens and repredicted condition tokens.

  Returns:
    Dict mapping str to [batch, sequence_length].
  """
  assert condition_mode in (None, 'prefix', 'cond->input', 'input->input',
                            'cond->cond')
  mask_ratio = jnp.clip(mask_ratio, 1e-6, 1)
  num_masked_limits = merge_limits(num_masked_limits, (1, None))  # pytype: disable=wrong-arg-types  # jax-ndarray

  if mask_score is None:
    # Uniformly generates the mask score.
    mask_score = jax.random.uniform(rng, shape=batch['inputs'].shape)

  inputs = batch['inputs']
  num_maskable = np.prod(batch['inputs'].shape[1:])
  if condition_mode in ('input->input', 'cond->cond'):
    # Fixed condition tokens
    mask_score = jnp.where(batch['cond_mask'], NEVER_MASK, mask_score)
    maskable = jnp.broadcast_to(~batch['cond_mask'], batch['inputs'].shape)
    num_maskable = maskable.sum(axis=(1, 2, 3))
    num_masked_limits = merge_limits(num_masked_limits, (None, num_maskable))
    if condition_mode == 'cond->cond':
      inputs = jnp.where(batch['cond_mask'], batch['cond_inputs'], inputs)

  sorted_mask_score = jnp.sort(batch_flatten(mask_score), axis=-1)

  # Number of tokens to mask.
  num_masked = jnp.floor(num_maskable * mask_ratio).astype(jnp.int32)
  num_masked = jnp.clip(num_masked, *num_masked_limits)
  # Obtains the cutoff score for the mask lens.
  cut_off = jnp.take_along_axis(
      sorted_mask_score, (num_masked - 1)[:, None], axis=-1)[:, :, None, None]
  # Scores smaller than the cutoff will be masked.
  should_mask = jnp.where(mask_score <= cut_off, 1., 0.)

  # Tokens to replace at masked positions
  if condition_mode == 'cond->input':
    replace_tokens = jnp.where(batch['cond_mask'], batch['cond_inputs'],
                               MASK_TOKEN)
  else:
    replace_tokens = MASK_TOKEN
  # Only replace positions where `should_mask`
  masked_inputs = jnp.where(should_mask, replace_tokens, inputs)
  uncond_masked_inputs = jnp.where(should_mask, MASK_TOKEN, inputs)
  if condition_mode == 'cond->input':
    segment_ids = jnp.where(
        jnp.logical_and(batch['cond_mask'], should_mask), condition_segment_id,
        0)
  else:
    segment_ids = jnp.zeros_like(masked_inputs)

  if weight_mode is None:
    weight_mode = 'mask+refine+recons'
  elif isinstance(weight_mode, bool):
    weight_mode = 'mask+refine' if weight_mode else 'mask+refine+recons'

  if weight_mode == 'mask+refine+recons':
    if condition_mode in ('input->input', 'cond->cond'):
      weights = maskable.astype(jnp.float32)
    else:
      weights = jnp.ones_like(should_mask)
  elif weight_mode == 'mask+refine':
    weights = should_mask
  else:
    raise NotImplementedError(f'Unknown weight mode: {weight_mode}')

  assert masked_inputs.shape == inputs.shape
  return {
      'masked_inputs': masked_inputs,
      'uncond_masked_inputs': uncond_masked_inputs,
      'segment_ids': segment_ids,
      'targets': inputs,
      'weights': weights,
  }


def _average_pooling_3d_in_place(
    value: jnp.ndarray,
    poolable: jnp.ndarray,
    window_size: tuple[int, int, int],
):
  """Average pooling 3D on poolable locations while keeping original shape."""
  bs, t, h, w = value.shape
  t_b, h_b, w_b = window_size
  grouped_shape = (bs, t // t_b, t_b, h // h_b, h_b, w // w_b, w_b)
  value = jnp.reshape(value, grouped_shape)
  poolable = jnp.reshape(poolable, grouped_shape)

  value_avg = jnp.mean(value, axis=(2, 4, 6), keepdims=True, where=poolable)
  tiles = (1, 1, t_b, 1, h_b, 1, w_b)
  value_avg = jnp.tile(value_avg, tiles)
  value = jnp.where(poolable, value_avg, value)

  value = jnp.reshape(value, (bs, t, h, w))
  return value


def random_block_mask(
    batch: TokenDict,
    rng: Optional[jnp.ndarray],
    mask_ratio: jnp.ndarray,
    *,
    condition_mode: Optional[str] = None,
    mask_score: Optional[jnp.ndarray] = None,
    num_masked_limits: Limit = (None, None),
    condition_segment_id: int = 1,
    weight_mode: Optional[str] = None,
    block_shape: tuple[int, int, int] = (1, 1, 1),
) -> TokenDict:
  """Random block mask on 3D input tokens.

  Block masking enforced by setting the same mask score for each block.
  Given a block shape, this function iterates over all possible offsets of
  the block and perform block average pooling on the mask score.

  Args:
    batch: Batch containing inputs/[cond_inputs/cond_mask], see random_mask.
    rng: PRNGKey.
    mask_ratio: Proportion of maskable tokens to mask out.
    condition_mode: Mode to handle condition tokens, see random_mask.
    mask_score: Optionally provided mask score, see random_mask.
    num_masked_limits: Optional limits on the number of masked tokens.
    condition_segment_id: Segment id for condition tokens.
    weight_mode: Mode to apply loss weights, see random_mask.
    block_shape: 3-tuple of block shape.

  Returns:
    Dict mapping str to [batch, sequence_length].
  """
  if mask_score is None:
    rng, subrng = jax.random.split(rng)
    mask_score = jax.random.uniform(subrng, shape=batch['inputs'].shape)

  # Add padding for block sampling
  t_p, h_p, w_p = block_shape[0] - 1, block_shape[1] - 1, block_shape[2] - 1
  padding = ((0, 0), (t_p, t_p), (h_p, h_p), (w_p, w_p))
  mask_score = jnp.pad(mask_score, padding, 'edge')
  # Build block offsets
  offsets = jnp.stack(
      jnp.meshgrid(*[jnp.arange(i) for i in (1, *block_shape)]),
      axis=-1).reshape(-1, 1 + len(block_shape))
  rng, subrng = jax.random.split(rng)
  offsets = jax.random.permutation(subrng, offsets)

  def cond_fn(state):
    i, _ = state
    return i < offsets.shape[0]

  def body_fn(state):
    i, score = state
    offset = offsets[i]
    # At each offset, perform block average pooling on the score.
    cur_score = jax.lax.dynamic_slice(score, offset, batch['inputs'].shape)
    cur_score = _average_pooling_3d_in_place(cur_score, cur_score != NEVER_MASK,
                                             block_shape)
    score = jax.lax.dynamic_update_slice(score, cur_score, offset)
    return i + 1, score

  mask_score = jax.lax.while_loop(cond_fn, body_fn, (0, mask_score))[1]

  # Remove padding
  mask_score = mask_score[:, t_p:-t_p if t_p > 0 else None,
                          h_p:-h_p if h_p > 0 else None,
                          w_p:-w_p if w_p > 0 else None]
  if block_shape != (1, 1, 1):
    rng, subrng = jax.random.split(rng)
    # Add different small values to break tie.
    mask_score = mask_score + 1e-3 * jax.random.uniform(
        subrng, shape=mask_score.shape)

  return random_mask(
      batch,
      rng,
      mask_ratio,
      condition_mode=condition_mode,
      mask_score=mask_score,
      num_masked_limits=num_masked_limits,
      condition_segment_id=condition_segment_id,
      weight_mode=weight_mode)


def frame_by_frame_mask(
    batch: TokenDict,
    rng: jnp.ndarray,
    mask_ratio: jnp.ndarray,
    *,
    condition_mode: Optional[str] = None,
    mask_score: Optional[jnp.ndarray] = None,
    num_masked_limits: Limit = (None, None),
    condition_segment_id: int = 1,
    weight_mode: Optional[str] = None,
    block_shape: tuple[int, int, int] = (1, 1, 1),
) -> TokenDict:
  """Frame-by-frame mask on 3D input tokens.

  Args:
    batch: Batch containing inputs/[cond_inputs/cond_mask], see random_mask.
    rng: PRNGKey.
    mask_ratio: Proportion of tokens to mask out, excluding condition tokens.
    condition_mode: Mode to handle condition tokens, see random_mask.
    mask_score: Optionally provided mask score, see random_mask.
    num_masked_limits: Optional limits on the number of masked tokens.
    condition_segment_id: Segment id for condition tokens.
    weight_mode: Mode to apply loss weights, see random_mask.
    block_shape: 3-tuple of block shape.

  Returns:
    Dict mapping str to [batch, sequence_length].
  """
  if mask_score is None:
    rng, subrng = jax.random.split(rng)
    mask_score = jax.random.uniform(subrng, shape=batch['inputs'].shape)

  total_frames = batch['inputs'].shape[1]
  frame_index = jnp.arange(total_frames)
  mask_score = normalize_score(mask_score) - frame_index[None, :, None, None]

  token_dict = random_block_mask(
      batch,
      rng,
      mask_ratio,
      condition_mode=condition_mode,
      mask_score=mask_score,
      num_masked_limits=num_masked_limits,
      condition_segment_id=condition_segment_id,
      weight_mode=weight_mode,
      block_shape=block_shape)
  return token_dict


def lm_tokens(ids: jnp.ndarray, bos: int) -> TokenDict:
  # inputs have the same shape as ids but they are shifted to the right by
  # prepending a bos token.
  inputs = jnp.concatenate(
      [jnp.full((ids.shape[0], 1), bos), ids,], axis=-1)[:, :-1]
  return {
      'inputs': inputs,
      'targets': ids,
      # pay attentio to the inputs with padding tokens,
      'weights': jnp.ones(inputs.shape, jnp.float32),
  }
