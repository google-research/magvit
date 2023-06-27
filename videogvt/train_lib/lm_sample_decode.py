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

"""Fast sampling-based decoding routines for autoregressive generation."""
from typing import Callable, Dict

import flax
import jax
from jax import lax
import jax.numpy as jnp
from videogvt.train_lib import sampling


@flax.struct.dataclass
class State:
  """Holds decoding state data."""
  step: jnp.ndarray  # scalar int32: current decoding iteration
  rng: jnp.ndarray  # Sampling random state.
  outputs: jnp.ndarray
  cache: Dict[str, jnp.array]
  score: jnp.ndarray


def state_init(inputs: jnp.ndarray,
               cache: Dict[str, jnp.array],
               rng: jnp.ndarray):
  """Initializes the decoding state data structure."""
  # assume each example has the same conditional lengths
  step = jnp.array(0)
  outputs = jnp.zeros((inputs.shape[0], inputs.shape[1] + 1), inputs.dtype)
  outputs = outputs.at[:, :inputs.shape[1]].set(inputs)
  score = jnp.zeros((inputs.shape[0],), jnp.float32)
  return State(
      step=step,
      rng=rng,
      outputs=outputs,
      cache=cache,
      score=score)


def decode(inputs: jnp.ndarray,
           prefix_lengths: jnp.ndarray,
           cache: Dict[str, jnp.ndarray],
           decode_fn: Callable[[jnp.ndarray, Dict[str, jnp.ndarray],
                                jnp.ndarray], jnp.ndarray],
           rng: jnp.ndarray,
           sampling_topk: int = 0,
           sampling_topp: float = 0,
           sampling_temperature: float = 1.0):
  """Fast decoding for bert iterative generation.

  In the decoding alogrithm, we take iterations to refine them.

  Args:
    inputs: array: [batch, T, H, W] int32 input masked tokens.
    prefix_lengths: [batch] for the length of prefix inputs.
    cache: cached states for self-attention.
    decode_fn: decoder function taking a batch of masked 3D tokens and returning
      predicted 3D tokens.
    rng: jnp.DeviceArray: sampling random state.
    sampling_topk: int: only the top-k logits will be considered to sample next
      token. If sampling_topk is zero, sample from full logits.
    sampling_topp: float: the smallest number of logits whose cumulative sum of
      probs adds up to topp. The next token will be sampled from these tokens.
      If zero, sample from full logits.
    sampling_temperature: temperature to control the randomness of sampling.

  Returns:
     Tuple of:
       [batch_size, max_decode_len] layout sequences
  """
  # Initializes state
  max_decode_step = inputs.shape[-1]
  init_state = state_init(inputs, cache, rng)

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    return state.step < max_decode_step

  def loop_body_fn(state):
    """Beam search loop state update function."""
    rng = state.rng
    step = state.step
    cache = state.cache
    outputs = state.outputs

    logits, new_cache = decode_fn(
        outputs[:, step, None],
        cache,
        step >= prefix_lengths - 1)
    # Samples predicted tokens at current step.
    new_rng, sample_rng = jax.random.split(rng, 2)
    cur_tokens = sampling.sampling(
        logits,
        sample_rng,
        topk=sampling_topk,
        topp=sampling_topp,
        temperature=sampling_temperature)
    new_ids = jnp.where(step < prefix_lengths - 1,
                        outputs[:, step+1, None],
                        cur_tokens)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    cur_tokens_log_probs = jnp.take_along_axis(
        jnp.reshape(log_probs, (log_probs.shape[0], -1)),
        cur_tokens,
        axis=-1)
    cur_log_probs = jnp.where(
        step < prefix_lengths - 1,
        jnp.ones((log_probs.shape[0], 1), log_probs.dtype),
        cur_tokens_log_probs)
    new_outputs = outputs.at[:, step+1].set(new_ids[:, 0])
    new_score = state.score + jnp.squeeze(cur_log_probs, -1)
    return State(
        step=step+1,
        rng=new_rng,
        outputs=new_outputs,
        cache=new_cache,
        score=new_score)
  final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  return final_state.outputs[:, 1:]
