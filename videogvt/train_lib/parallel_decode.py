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

"""Fast decoding routines for non-autoregressive generation."""
from typing import Callable

import flax
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from videogvt.train_lib import mask_schedule
from videogvt.train_lib import mask_utils
from videogvt.train_lib import sampling


@flax.struct.dataclass
class State:
  """Holds decoding state data."""
  cur_iter: jnp.ndarray  # scalar int32: current decoding iteration
  masked_tokens: jnp.ndarray  # int32 [batch, T, H, W]: current masked 3D tokens
  uncond_masked_tokens: jnp.ndarray  # int32 [batch, T, H, W]: masked 3D tokens
  # without interior condition mask for classifier-free guidance sampling.
  segment_ids: jnp.ndarray  # int32 [batch, T, H, W]: current token segment ids
  unknown_mask: jnp.ndarray  # bool [batch, T, H, W]: True for tokens to predict
  rng: jnp.ndarray  # Sampling random state.
  # Current full 3D tokens, or full 3D tokens from all iterations.
  # int32 [batch, 1, T, H, W] or [batch, num_iter, T, H, W]
  full_tokens: jnp.ndarray


def state_init(inputs: jnp.ndarray,
               rng: jnp.ndarray,
               mask_fn: mask_utils.MaskFn,
               num_iter: int,
               keep_intermediates: bool = False):
  """Initializes the decoding state data structure."""
  cur_iter0 = jnp.array(0)
  token_dict = mask_fn(inputs, rng, jnp.ones((1,)), weight_mode='mask+refine')
  masked_tokens = token_dict['masked_inputs']
  uncond_masked_tokens = token_dict['uncond_masked_inputs']
  segment_ids = token_dict['segment_ids']
  unknown_mask = token_dict['weights'] == 1.
  if keep_intermediates:
    full_tokens0 = jnp.empty_like(
        inputs, shape=(inputs.shape[0], num_iter, *inputs.shape[1:]))
  else:
    full_tokens0 = jnp.empty_like(inputs[:, None])
  return State(
      cur_iter=cur_iter0,
      masked_tokens=masked_tokens,
      uncond_masked_tokens=uncond_masked_tokens,
      segment_ids=segment_ids,
      unknown_mask=unknown_mask,
      rng=rng,
      full_tokens=full_tokens0)


def decode(inputs: jnp.ndarray,
           decode_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                               jnp.ndarray],
           mask_fn: mask_utils.MaskFn,
           rng: jnp.ndarray,
           num_iter: int = 12,
           sampling_topk: int = 0,
           sampling_topp: float = 0.,
           sampling_temperature: float = 1.0,
           mask_temperature: float = 1.0,
           mask_scheduling_method: str = 'cosine',
           keep_intermediates: bool = False):
  """Fast decoding for bert iterative generation.

  In the decoding alogrithm, we take iterations to refine them.

  Args:
    inputs: array: [batch, T, H, W] int32 input masked tokens.
    decode_fn: decoder function taking a batch of masked 3D tokens with segment
      ids and returning predicted 3D tokens.
    mask_fn: mask function taking a batch of full 3D tokens with confidence
      scores and returning masked 3D tokens with segment ids according to the
      current mask ratio.
    rng: jnp.DeviceArray: sampling random state.
    num_iter: number of decoding iterations.
    sampling_topk: int: only the top-k logits will be considered to sample next
      token. If sampling_topk is zero, sample from full logits.
    sampling_topp: float: the smallest number of logits whose cumulative sum of
      probs adds up to topp. The next token will be sampled from these tokens.
      If zero, sample from full logits.
    sampling_temperature: temperature to control the randomness of sampling.
    mask_temperature: temperature to control the randomness of masking.
    mask_scheduling_method: masking method string. See mask_schedule.py for
      details.
    keep_intermediates: whether to keep intermediate decoding results.

  Returns:
     Tuple of:
       [batch_size, max_decode_len] layout sequences
  """
  num_tokens = np.prod(inputs.shape[1:])
  # Initializes state
  init_state = state_init(inputs, rng, mask_fn, num_iter, keep_intermediates)

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    return state.cur_iter < num_iter

  def loop_body_fn(state):
    """Beam search loop state update function."""
    rng = state.rng
    step = state.cur_iter
    prev_tokens = state.masked_tokens
    prev_uncond_masked_tokens = state.uncond_masked_tokens
    prev_segment_ids = state.segment_ids
    unknown_mask = state.unknown_mask

    # Calls model on current seqs to get next-iteration seqs.
    logits = decode_fn(prev_tokens, prev_segment_ids,
                       prev_uncond_masked_tokens)  # [batch, T, H, W, vocab]
    # Samples predicted tokens at current step.
    rng, sample_rng = jax.random.split(rng, 2)
    cur_tokens = sampling.sampling(
        logits,
        sample_rng,
        topk=sampling_topk,
        topp=sampling_topp,
        temperature=sampling_temperature)
    # Just updates the masked tokens.
    num_unknown = jnp.sum(unknown_mask, axis=(1, 2, 3))
    cur_tokens = jnp.where(unknown_mask, cur_tokens, prev_tokens)
    # Updates full_seqs with the current sampled_tokens.
    if keep_intermediates:
      full_tokens = state.full_seqs.at[:, step].set(cur_tokens)
    else:
      full_tokens = cur_tokens[:, None]

    # Computes the probabilities of each selected tokens.
    probs = jax.nn.softmax(logits, axis=-1)
    cur_probs = jnp.take_along_axis(probs, cur_tokens[..., None],
                                    -1)[..., 0]  # [batch, T, H, W]
    # Ignores the tokens given in the input by overwriting their confidence.
    cur_probs = jnp.where(unknown_mask, cur_probs, mask_utils.NEVER_MASK)
    # Mask ratio for the next round.
    progress = (step + 1) / num_iter
    mask_ratio = mask_schedule.schedule(progress, num_tokens,
                                        mask_scheduling_method)[None]

    # Adds noise to mask score for randomness
    rng, score_rng, mask_rng = jax.random.split(rng, 3)
    mask_score = jnp.log(cur_probs) + mask_temperature * (
        1. - progress) * jax.random.gumbel(score_rng, cur_probs.shape)
    # Mask tokens
    token_dict = mask_fn(
        cur_tokens,
        mask_rng,
        mask_ratio,
        weight_mode='mask+refine',
        mask_score=mask_score,
        num_masked_limits=(1, num_unknown - 1))
    masked_tokens = token_dict['masked_inputs']
    uncond_masked_tokens = token_dict['uncond_masked_inputs']
    segment_ids = token_dict['segment_ids']
    unknown_mask = token_dict['weights'] == 1.
    return State(
        cur_iter=state.cur_iter + 1,
        masked_tokens=masked_tokens,
        uncond_masked_tokens=uncond_masked_tokens,
        segment_ids=segment_ids,
        unknown_mask=unknown_mask,
        rng=rng,
        full_tokens=full_tokens)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  return final_state.full_tokens
