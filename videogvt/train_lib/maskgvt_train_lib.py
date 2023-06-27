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

"""Lib for the MaskGVT trainer."""
import functools
from typing import Any, Callable, Dict, Optional

from absl import logging
from clu import metrics
from clu import parameter_overview
import einops
import flax
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.train_lib import train_utils as scenic_train_utils
from videogvt.models import simplified_bert
from videogvt.train_lib import lm_sample_decode
from videogvt.train_lib import mask_schedule
from videogvt.train_lib import mask_utils
from videogvt.train_lib import parallel_decode
from videogvt.train_lib import task_registry
from videogvt.train_lib import train_state_manager
from videogvt.train_lib import train_utils


Batch = train_utils.Batch
TrainState = train_state_manager.ScenicTrainState
TokenizerFn = Callable[[jnp.ndarray], jnp.ndarray]


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """metrics for transformers."""
  total_loss: metrics.Average.from_output('total_loss')
  token_loss: metrics.Average.from_output('token_loss')
  l2_loss: metrics.Average.from_output('l2_loss')
  mask_ratio: metrics.LastValue.from_output('mask_ratio')
  grad_norm: metrics.Average.from_output('grad_norm')


def sync_vq_model_config(config: ml_collections.ConfigDict,
                         vq_config: ml_collections.ConfigDict):
  assert config.get('vq_codebook_size', -1) < 0
  with config.unlocked():
    config.vq_codebook_size = vq_config.vqvae.codebook_size
  quantize_steps = config.get('vqvae', dict()).get('quantize_steps', None)
  if quantize_steps:
    with vq_config.unlocked():
      vq_config.vqvae.quantize_steps = quantize_steps


def sync_lowres_config(config: ml_collections.ConfigDict,
                       lowres_config: ml_collections.ConfigDict):
  with lowres_config.unlocked():
    lowres_config.batch_size = config.batch_size
    lowres_config.eval_batch_size = config.eval_batch_size


def sync_total_seq_length(config: ml_collections.ConfigDict,
                          total_seq_length: int):
  assert config.get('total_seq_length', -1) < 0
  with config.unlocked():
    config.total_seq_length = total_seq_length


def get_vq_codebook_size(config: ml_collections.ConfigDict):
  assert config.vq_codebook_size > 0
  return config.vq_codebook_size


def get_num_classes(config: ml_collections.ConfigDict):
  return config.dataset_configs.num_classes + 1  # add UNCOND


def get_num_special_tokens(config: ml_collections.ConfigDict):
  """Gets the number of special tokens."""
  if not config.get('multi_task', False):
    return 1  # MASK
  num_slots = 16  # Placeholder for new tasks
  assert -min(task_registry.SpecialTokens) < num_slots
  return num_slots


def get_latent_seq_len(config: ml_collections.ConfigDict):
  """Returns the sequence length for generation, excluding condition tokens."""
  latent_seq_len = np.prod(config.transformer.latent_shape)
  return latent_seq_len


def get_bert_model(config: ml_collections.ConfigDict):
  """Build the BERT model."""
  variant = config.transformer.get('variant', 'simplified_bert')
  if variant == 'simplified_bert':
    model_class = simplified_bert.Bert
  else:
    raise NotImplementedError(f'Unknown transformer variant: {variant}')
  vq_codebook_size = get_vq_codebook_size(config)
  num_classes = get_num_classes(config)
  num_special_tokens = get_num_special_tokens(config)
  vocab_size = vq_codebook_size + num_classes + num_special_tokens
  num_segments = 2 if config.transformer.get(
      'use_condition_segment', False) else 0
  return model_class(
      vocab_size=vocab_size,
      hidden_size=config.transformer.hidden_size,
      num_hidden_layers=config.transformer.num_layers,
      num_attention_heads=config.transformer.num_heads,
      intermediate_size=config.transformer.mlp_dim,
      hidden_dropout_prob=config.transformer.dropout_rate,
      attention_probs_dropout_prob=config.transformer.attention_dropout_rate,
      max_position_embeddings=config.total_seq_length,
      num_segments=num_segments,
      pad_token_id=task_registry.SpecialTokens.PAD)


def get_class_tokens(batch: Batch, config: ml_collections.ConfigDict,
                     task_config: ml_collections.ConfigDict,
                     return_uncond_token: bool = False):
  """Gets the class tokens for a batch."""
  offset = get_vq_codebook_size(config)
  if task_config.get('class_conditional', False) and not return_uncond_token:
    class_token = batch['label'] + offset
  else:
    uncond_token = config.dataset_configs.num_classes + offset
    class_token = jnp.full((batch['inputs'].shape[0], 1), uncond_token,
                           jnp.int32)
  return class_token


def generate_masked_tokens(batch: Batch,
                           rng: Optional[jnp.ndarray],
                           mask_ratio: jnp.ndarray,
                           task: task_registry.Task,
                           config: ml_collections.ConfigDict,
                           *,
                           no_latent: bool = False):
  """Generate masked input tokens with condition tokens.

  Args:
    batch: A single batch of data, where batch['inputs'] is latent code of shape
      [device_bs, l_t, l_h, l_w] in jnp.int32, optionally batch['cond_inputs']
      is latent code like batch['inputs'] and batch['label'] is class id of
      shape [device_bs, 1] in jnp.int32.
    rng: Optional PRNGKey for task.mask_fn.
    mask_ratio: of shape [1] or shape [device_bs] where values are in (0,1].
    task: current task.
    config: model config.
    no_latent: whether only returning prefix tokens without latent tokens.

  Returns:
    Dictionary of the masked and target token sequence with the class info.
  """
  batch_size = batch['inputs'].shape[0]
  tokens_list = []

  # Task condition token
  if config.get('multi_task', False):
    task_tokens = jnp.full((batch_size, 1), task.task_token, jnp.int32)
    if any([task.startswith('full_generation') for task in config.tasks]):
      uncond_task_tokens = jnp.full(
          (batch_size, 1), task_registry.SpecialTokens.TASK_FULL_GENERATION,
          jnp.int32)
    else:
      uncond_task_tokens = task_tokens
    tokens_list.append(
        mask_utils.no_mask({
            'inputs': task_tokens,
            'uncond_inputs': uncond_task_tokens
        }))

  # Class condition token
  class_tokens = get_class_tokens(batch, config, task.config)
  uncond_class_tokens = get_class_tokens(
      batch, config, task.config, return_uncond_token=True)
  tokens_list.append(
      mask_utils.no_mask({
          'inputs': class_tokens,
          'uncond_inputs': uncond_class_tokens
      }))

  # Frame condition tokens, deprecated, only for single frame prediction task
  if (task.task_id == 'frame_prediction' and
      task.config.get('condition_mode') == 'prefix'):
    num_frames = config.frame_prediction.cond_latent_frames
    cond_latents = batch.pop('cond_inputs')[:, :num_frames]
    tokens_list.append(mask_utils.no_mask({'inputs': cond_latents}))
    sep_token_ids = jnp.full((batch_size, 1), task_registry.SpecialTokens.SEP,
                             jnp.int32)
    tokens_list.append(mask_utils.no_mask({'inputs': sep_token_ids}))
    batch.pop('cond_mask')

  # Latent tokens
  if not no_latent:
    latent_tokens = task.mask_fn(batch, rng, mask_ratio)
    tokens_list.append(latent_tokens)

  all_tokens = mask_utils.concat_tokens(*tokens_list)

  return all_tokens


def generate_tokens_for_lm(batch: Batch,
                           task: task_registry.Task,
                           config: ml_collections.ConfigDict,):
  """Generate input tokens for language model.

  Args:
    batch: A single batch of data, where batch['inputs'] is latent code of shape
      [device_bs, l_t, l_h, l_w] in jnp.int32, batch['label'] is class id of
      shape [device_bs, 1] in jnp.int32 when config.class_conditional is True,
      and batch['cond_inputs'] is latent code like batch['inputs'] when
      config.frame_conditional is True.
    task: current task.
    config: model config.

  Returns:
    Dictionary of the masked and target token sequence with the class info.
  """
  batch_size = batch['inputs'].shape[0]
  tokens_list = []

  assert not config.get('multi_task', False)
  # Task condition token
  if config.get('multi_task', False):
    task_tokens = jnp.full((batch_size, 1), task.task_token, jnp.int32)
    tokens_list.append(mask_utils.no_mask({'inputs': task_tokens}))

  # Class condition token
  class_tokens = get_class_tokens(batch, config, task.config)
  tokens_list.append(mask_utils.no_mask({'inputs': class_tokens}))

  # Frame condition tokens
  if (task.task_id == 'frame_prediction' and task.config.get(
      'condition_mode') == 'prefix'):
    num_frames = config.frame_prediction.cond_latent_frames
    cond_latents = batch.pop('cond_inputs')[:, :num_frames]
    tokens_list.append(mask_utils.no_mask({'inputs': cond_latents}))
    sep_token_ids = jnp.full((batch_size, 1), task_registry.SpecialTokens.SEP,
                             jnp.int32)
    tokens_list.append(mask_utils.no_mask({'inputs': sep_token_ids}))
    batch.pop('cond_mask')

  # Latent tokens
  latent_tokens = mask_utils.no_mask(batch)
  tokens_list.append(latent_tokens)
  all_tokens = mask_utils.concat_tokens(*tokens_list)
  # no mask tokens in lm, so here use mask token as bos token.
  all_tokens = mask_utils.lm_tokens(all_tokens['masked_inputs'],
                                    task_registry.SpecialTokens.MASK)
  cond_tokens = mask_utils.concat_tokens(*(tokens_list[:-1]))
  # 1 for bos
  all_tokens['cond_lengths'] = cond_tokens['masked_inputs'].shape[-1] + 1
  return all_tokens


def load_vq_tokenizer(config: ml_collections.ConfigDict,
                      *,
                      tokenizer_only: bool = False,
                      log_parameters: bool = True) -> Dict[str, TokenizerFn]:
  """Loads the pretrained the VQ model as tokenizer."""
  if config.get('debug_pseudo_vq', False):
    return get_pseudo_tokenizer_fn(config)
  vq_params, vq_config = load_vq_model_params(config)
  sync_vq_model_config(config, vq_config)
  vq_model = train_utils.get_vq_model(vq_config)['generator']
  if log_parameters:
    logging.info('logging vq_model parameters')
    parameter_overview.log_parameter_overview(vq_params)
  enc_params, _ = vq_params.pop('decoder')
  tokenizer_fn = functools.partial(
      vq_model.apply, {'params': enc_params}, method=vq_model.encode_to_indices)
  if tokenizer_only:
    return {'tokenizer': tokenizer_fn}
  dec_params, _ = vq_params.pop('encoder')
  detokenizer_fn = functools.partial(
      vq_model.apply, {'params': dec_params},
      method=vq_model.decode_from_indices)
  return {'tokenizer': tokenizer_fn, 'detokenizer': detokenizer_fn}


def get_pseudo_tokenizer_fn(
    config: ml_collections.ConfigDict) -> Dict[str, TokenizerFn]:
  """For the unit tests."""
  l_t, l_h, l_w = config.transformer.latent_shape
  image_size = config.image_size
  if isinstance(image_size, int):
    height = width = image_size
  else:
    height, width = image_size
  r_t = config.dataset_configs.num_frames // l_t
  r_h = height // l_h
  r_w = width // l_w

  codebook_size = config.vq_codebook_size
  def tokenizer_fn(x: jnp.ndarray):
    return (x * codebook_size).astype(jnp.int32)[:, :l_t, :l_h, :l_w, 0]
  def detokenizer_fn(x: jnp.ndarray):
    x = (x.astype(jnp.float32) / codebook_size)[..., None]
    return jnp.tile(x, (1, r_t, r_h, r_w, 3))
  return {'tokenizer': tokenizer_fn, 'detokenizer': detokenizer_fn}


def load_vq_model_params(config: ml_collections.ConfigDict):
  """Load the pretrained VQGAN model."""
  vq_checkpoint_path = config.vq_model_from.get('checkpoint_path')
  vq_config = config.vq_model_from.get('config')
  if vq_config is not None and vq_checkpoint_path is None:
    vq_checkpoint_path = vq_config.eval_from.get('checkpoint_path')
  vq_train_state = checkpoints.restore_checkpoint(vq_checkpoint_path, None)
  vq_params = flax.core.freeze(vq_train_state['ema_params'])
  return vq_params, vq_config


def get_task_idx_sequence(task_weights: np.ndarray,
                          rng: jnp.ndarray) -> np.ndarray:
  normalizer = 100 / np.minimum(1, task_weights.min())
  task_weights = np.round(task_weights * normalizer).astype(np.int32)
  task_idx_seq = np.repeat(np.arange(task_weights.shape[0]), task_weights)
  task_idx_seq = jax.device_get(jax.random.permutation(rng, task_idx_seq))
  return task_idx_seq


def tokenize_with_cond(inputs, cond_fn, *, tokenizer_fn):
  """Run tokenizer on inputs and derived condition."""
  if cond_fn is not None:
    # Use cond latents from masked inputs.
    cond_dict = cond_fn(inputs)
    all_inputs = jnp.concatenate([inputs, cond_dict.pop('video')])
    all_tokens = tokenizer_fn(all_inputs)
    input_tokens, cond_tokens = jnp.split(all_tokens, 2)
    return {
        'inputs': input_tokens,
        'cond_inputs': cond_tokens,
        'cond_mask': cond_dict['latent_mask']
    }
  else:
    input_tokens = tokenizer_fn(inputs)
    return {'inputs': input_tokens}


def fast_decode(batch: Batch, prefix_token_dict: jnp.ndarray,
                mask_fn: mask_utils.MaskFn, rng: jnp.ndarray, variables,
                flax_model: nn.Module,
                inference_method: Optional[Callable[..., Any]],
                config: ml_collections.ConfigDict) -> jnp.ndarray:
  """Fast decoding method.

  Args:
    batch: Dictionary where batch['inputs'].shape=[batch, T, H, W] is the full
      tokens, and optionally batch['cond_inputs'] and batch['cond_mask'] in a
      broadcastable shape to batch['inputs'] are the condition tokens and masks.
    prefix_token_dict: Dictionary where prefix_token_dict['masked_inputs'].shape
      =[batch, cond_len] is the input prefix tokens and
      prefix_token_dict['segment_ids'] is segment ids in the same shape.
    mask_fn: mask function.
    rng: PRNGKey for decoding.
    variables: transformer model weights.
    flax_model: transformer model object.
    inference_method: optional method to inference transformer.
    config: model config.

  Returns:
    [batch, seq_len] decoded token sequence.
  """
  vq_codebook_size = get_vq_codebook_size(config)
  latent_seq_len = get_latent_seq_len(config)

  def decode_fn(cur_inputs: jnp.ndarray,
                segment_ids: jnp.ndarray,
                cur_uncond_inputs: jnp.ndarray | None = None) -> jnp.ndarray:
    del cur_uncond_inputs
    masked_inputs = mask_utils.concat_tokens(prefix_token_dict['masked_inputs'],
                                             cur_inputs)
    segment_ids = mask_utils.concat_tokens(prefix_token_dict['segment_ids'],
                                           segment_ids)
    model_inputs = [masked_inputs, segment_ids]
    if 'lowres_tokens' in batch:
      model_inputs.append(batch['lowres_tokens'])

    logits = flax_model.apply(
        variables,
        *model_inputs,
        method=inference_method,
        deterministic=True)
    logits = logits[:, -latent_seq_len:, :vq_codebook_size]
    logits = logits.reshape(*cur_inputs.shape, vq_codebook_size)
    return logits

  def decode_fn_with_guidance(cur_inputs: jnp.ndarray, segment_ids: jnp.ndarray,
                              cur_uncond_inputs: jnp.ndarray,
                              guidance_scale: float) -> jnp.ndarray:
    masked_inputs = mask_utils.concat_tokens(prefix_token_dict['masked_inputs'],
                                             cur_inputs)
    segment_ids = mask_utils.concat_tokens(prefix_token_dict['segment_ids'],
                                           segment_ids)
    model_inputs = [masked_inputs, segment_ids]
    if 'lowres_tokens' in batch:
      model_inputs.append(batch['lowres_tokens'])
    logits = flax_model.apply(
        variables,
        *model_inputs,
        method=inference_method,
        deterministic=True)
    logits = logits[:, -latent_seq_len:, :vq_codebook_size]
    uncond_inputs = mask_utils.concat_tokens(
        prefix_token_dict['uncond_masked_inputs'], cur_uncond_inputs
    )
    uncond_logits = flax_model.apply(
        variables,
        uncond_inputs,
        segment_ids,
        method=inference_method,
        deterministic=True)
    uncond_logits = uncond_logits[:, -latent_seq_len:, :vq_codebook_size]
    logits = logits + guidance_scale * (logits - uncond_logits)
    logits = logits.reshape(*cur_inputs.shape, vq_codebook_size)
    return logits

  def mask_fn_with_cond(cur_inputs: jnp.ndarray, *args,
                        **kwargs) -> mask_utils.TokenDict:
    cur_batch = {'inputs': cur_inputs}
    if 'cond_inputs' in batch:
      cur_batch['cond_inputs'] = batch['cond_inputs']
    if 'cond_mask' in batch:
      cur_batch['cond_mask'] = batch['cond_mask']
    return mask_fn(cur_batch, *args, **kwargs)

  guidance_scale = config.sampling.get('guidance_scale', 0.0)
  output_indices = parallel_decode.decode(
      batch['inputs'],
      decode_fn if guidance_scale == 0.0 else functools.partial(
          decode_fn_with_guidance, guidance_scale=guidance_scale),
      mask_fn_with_cond,
      rng,
      num_iter=config.sampling.mask_bins,
      sampling_topk=config.sampling.get('sampling_topk', 0),
      sampling_topp=config.sampling.get('sampling_topp', 0.),
      mask_temperature=config.sampling.choice_temperature,
      mask_scheduling_method=config.sampling.mask_scheduling_method)
  output_indices = jnp.array(output_indices[:, -1], dtype=jnp.int32)
  return output_indices


def lm_decode(seq: jnp.ndarray,
              prefix_lengths: jnp.ndarray,
              cache: Dict[str, jnp.ndarray],
              rng: jnp.ndarray,
              variables,
              flax_model: nn.Module,
              config: ml_collections.ConfigDict) -> jnp.ndarray:
  """Fast decoding method.

  Args:
    seq: [batch, cond_len] input tokens.
    prefix_lengths: [batch,] prefix lengths.
    cache: cached variables for self-attention layers.
    rng: PRNGKey for decoding.
    variables: transformer model weights.
    flax_model: transformer model object.
    config: model config.

  Returns:
    [batch, seq_len] decoded token sequence.
  """
  vq_codebook_size = get_vq_codebook_size(config)
  latent_seq_len = get_latent_seq_len(config)

  def decode_fn(inputs, cache, enable_vq_codebook=jnp.array(False, bool)):
    new_variables = dict(list(variables.items()) + [('cache', cache)])
    outputs = flax_model.apply(
        new_variables,
        inputs,
        mutable=['cache'],
        deterministic=True)
    # [bs, 1, vocab_size]
    logits, new_cache = outputs
    all_valid = jnp.ones((logits.shape[-1],), logits.dtype)
    vq_codebook_valid = jnp.concatenate([
        jnp.ones((vq_codebook_size,), logits.dtype),
        jnp.zeros((logits.shape[-1] - vq_codebook_size,), logits.dtype)
    ],
                                        axis=0)
    logit_valid = jnp.where(enable_vq_codebook, vq_codebook_valid, all_valid)
    logits = logits * jnp.broadcast_to(logit_valid, logits.shape) - 1000 * (
        1 - jnp.broadcast_to(logit_valid, logits.shape))
    return logits, new_cache['cache']
  output_indices = lm_sample_decode.decode(
      seq,
      prefix_lengths,
      cache,
      decode_fn,
      rng,
      sampling_topk=config.sampling.sampling_topk,
      sampling_topp=config.sampling.sampling_topp,
      sampling_temperature=config.sampling.sampling_temperature)
  output_indices = output_indices[:, -latent_seq_len:]
  return output_indices


def generate_mask_ratio(rng: jnp.ndarray,
                        config: ml_collections.ConfigDict,
                        local_bs: Optional[int] = None):
  if local_bs is None:
    local_bs = 1  # uniform for the entire batch
  ratio = jax.random.uniform(rng, [local_bs])
  return mask_schedule.schedule(
      ratio, get_latent_seq_len(config), method=config.mask_scheduling_method)


def get_optimizer(
    learning_rate_fn: Callable[[int], float],
    config: ml_collections.ConfigDict) -> optax.GradientTransformation:
  """Constructs the optimizer from the given HParams."""
  weight_decay = config.optimizer.get('weight_decay', 0.0)
  decay_list = ['kernel']
  no_decay_list = ['bias', 'scale', 'embedding', 'embed_pos']

  decay_param_mask = train_utils.flattened_traversal(
      lambda path, _: any(pn in path for pn in decay_list))
  no_decay_param_mask = train_utils.flattened_traversal(
      lambda path, _: any(pn in path for pn in no_decay_list))
  # using inject_hyperparams gets learning_rate but needs manual lr management.
  tx1 = optax.adamw(  # pytype: disable=wrong-arg-types  # numpy-scalars
      learning_rate=learning_rate_fn,
      b1=config.optimizer.beta1,
      b2=config.optimizer.beta2,
      weight_decay=weight_decay)

  tx2 = optax.adamw(  # pytype: disable=wrong-arg-types  # numpy-scalars
      learning_rate=learning_rate_fn,
      b1=config.optimizer.beta1,
      b2=config.optimizer.beta2,
      weight_decay=0.0)

  tx = optax.chain(
      optax.masked(tx1, decay_param_mask),
      optax.masked(tx2, no_decay_param_mask),
  )
  return tx


def get_standard_adamw_optimizer(
    learning_rate_fn: Callable[[int], float],
    config: ml_collections.ConfigDict) -> optax.GradientTransformation:
  """Constructs the optimizer from the given HParams."""
  weight_decay = config.optimizer.get('weight_decay', 0.0)
  tx = optax.adamw(  # pytype: disable=wrong-arg-types  # numpy-scalars
      learning_rate=learning_rate_fn,
      b1=config.optimizer.beta1,
      b2=config.optimizer.beta2,
      weight_decay=weight_decay)
  return tx


def sync_model_state_across_replicas(state: TrainState) -> TrainState:
  """Sync the model_state (simple averaging) across replicas."""
  return scenic_train_utils.sync_model_state_across_replicas(state)


def illustrate_predictions(masked_inputs: jnp.ndarray, logits: jnp.ndarray,
                           targets: jnp.ndarray,
                           config: ml_collections.ConfigDict) -> jnp.ndarray:
  """illustrate the predictions as an image."""
  latent_shape = config.transformer.latent_shape
  latent_seq_len = get_latent_seq_len(config)
  masks = masked_inputs == task_registry.SpecialTokens.MASK  # TODO(Lijun-Yu): fix
  masks_3d = jnp.reshape(masks[:, -latent_seq_len:], [-1, *latent_shape])
  tile_param = [*masks_3d.shape, 1]

  light_red = jnp.tile(jnp.array([255, 229, 231]), tile_param)  # unmasked wrong
  dark_red = jnp.tile(jnp.array([139, 0, 0]), tile_param)  # masked wrong

  light_green = jnp.tile(jnp.array([229, 255, 229]),
                         tile_param)  # unmasked right
  dark_green = jnp.tile(jnp.array([21, 71, 52]), tile_param)  # masked right

  preds = jnp.argmax(logits, axis=-1)
  correct = jnp.equal(preds, targets)
  correct = jnp.reshape(correct[:, -latent_seq_len:], masks_3d.shape)

  not_correct = jnp.logical_not(correct)
  not_masks_3d = jnp.logical_not(masks_3d)

  a = jnp.logical_and(masks_3d, correct)
  b = jnp.logical_and(masks_3d, not_correct)
  c = jnp.logical_and(not_masks_3d, correct)
  # d = jnp.logical_and(not_masks_3d, not_correct)

  result_image = light_red
  result_image = jnp.where(a[..., None], dark_green, result_image)
  result_image = jnp.where(b[..., None], dark_red, result_image)
  result_image = jnp.where(c[..., None], light_green, result_image)
  return result_image / 255.


def draw_video_boundary(x: jnp.ndarray) -> jnp.ndarray:
  """Pads the frame boundary with black pixels. x.shape=[bs, lt, lh, lw, 3]."""
  assert x.ndim == 5
  width = 1
  x = jnp.pad(
      x, [(0, 0), (0, 0), (0, 0), (width, width), (0, 0)],
      mode='constant',
      constant_values=0)
  x1 = einops.rearrange(x, 'device_bs t h w c -> device_bs h (t w) c')
  x1 = jnp.pad(
      x1, [(0, 0), (width, width), (0, 0), (0, 0)],
      mode='constant',
      constant_values=0)
  x2 = jnp.concatenate(x1, axis=0)
  return x2


def expand_pos_embedding(tree_old, tree_new, *, latent_seq_len):
  """Add condition token positions before latent sequence."""
  cur = tree_new['BertEmbed_0']['position_embeddings']['embedding']
  loaded = tree_old['BertEmbed_0']['position_embeddings']['embedding']
  assert cur.shape[0] >= loaded.shape[0]
  # Prefix positions: class or uncond token
  prefix_length = loaded.shape[0] - latent_seq_len
  cur = cur.at[:prefix_length].set(loaded[:prefix_length])
  # Latent sequence
  suffix_length = latent_seq_len
  cur = cur.at[-suffix_length:].set(loaded[-suffix_length:])
  tree_old['BertEmbed_0']['position_embeddings']['embedding'] = cur
  return tree_old


def expand_word_embedding(tree_old, tree_new, *, vq_codebook_size, num_classes):
  """Add task tokens before current special tokens and after class tokens."""
  cur_embed = tree_new['BertEmbed_0']['word_embeddings']['embedding']
  cur_bias = tree_new['BertMlmLayer_0']['mlm_bias']['bias']
  loaded_embed = tree_old['BertEmbed_0']['word_embeddings']['embedding']
  loaded_bias = tree_old['BertMlmLayer_0']['mlm_bias']['bias']
  assert cur_embed.shape[0] >= loaded_embed.shape[0]
  # Latent and class tokens
  prefix_length = vq_codebook_size + num_classes
  cur_embed = cur_embed.at[:prefix_length].set(loaded_embed[:prefix_length])
  cur_bias = cur_bias.at[:prefix_length].set(loaded_bias[:prefix_length])
  # Other existing special tokens
  suffix_length = loaded_embed.shape[1] - prefix_length
  cur_embed = cur_embed.at[-suffix_length:].set(loaded_embed[-suffix_length:])
  cur_bias = cur_bias.at[-suffix_length:].set(loaded_bias[-suffix_length:])
  tree_old['BertEmbed_0']['word_embeddings']['embedding'] = cur_embed
  tree_old['BertMlmLayer_0']['mlm_bias']['bias'] = cur_bias
  return tree_old


def init_from_pretrained_checkpoint(
    checkpoint_path: str,
    train_state: TrainState,
    expand_embedding: bool = False,
    config: Optional[ml_collections.ConfigDict] = None):
  """Initialize the train state with a pretrained checkpoint.

  First restores the checkpoint, then expands position embedding if needed.

  Args:
    checkpoint_path: Directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    expand_embedding: Whether expand embedding for condition tokens.
    config: Current model config.

  Returns:
    Training state and an int which is the current step.
  """
  restored_train_state = checkpoints.restore_checkpoint(checkpoint_path, None)
  if restored_train_state is None:
    raise ValueError('No checkpoint for the pretrained model is found in: '
                     f'{checkpoint_path}')
  restored_train_state = flax.core.freeze(restored_train_state)
  if 'optimizer' in restored_train_state:
    # Legacy train_state
    restored_params = restored_train_state['optimizer']['target']
  else:
    restored_params = restored_train_state['params']
  restored_model_state = restored_train_state.get('model_state')
  if expand_embedding:
    assert config is not None, ('config must be provided when '
                                'expand_embedding=True')
    latent_seq_len = get_latent_seq_len(config)
    vq_codebook_size = get_vq_codebook_size(config)
    num_classes = get_num_classes(config)
    restored_params = flax.core.unfreeze(restored_params)
    restored_params = expand_pos_embedding(
        restored_params, train_state.params, latent_seq_len=latent_seq_len)
    restored_params = expand_word_embedding(
        restored_params, train_state.params, vq_codebook_size=vq_codebook_size,
        num_classes=num_classes)
    restored_params = flax.core.freeze(restored_params)
  # pytype: disable=attribute-error
  new_train_state = train_state.replace(
      params=restored_params, model_state=restored_model_state)
  # pytype: enable=attribute-error
  return new_train_state
