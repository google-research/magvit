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

"""Multimodal Vector Quantization Public Interface."""
import functools
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from clu import parameter_overview
import flax
import flax.linen as nn
from flax.training import checkpoints
import jax.numpy as jnp
import ml_collections
from scenic.common_lib import export_utils
import tensorflow as tf
from tensorflow.io import gfile
from videogvt.train_lib import train_utils


# TODO(roadjiang): links these def in videogvt train_utils.
Batch = Dict[str, jnp.ndarray]
PyTree = Any
Batch = Dict[str, jnp.ndarray]
TokenizerFn = Callable[[jnp.ndarray], jnp.ndarray]


def get_mm_vq_model_and_params(
    config: ml_collections.ConfigDict) -> Tuple[nn.Module, PyTree]:
  """Loads a multimodal vector quantization model and pretrained parameters.

  Args:
    config: Model config file.

  Returns:
    A 2-tuple of a VQ model and the parameters for that model.
  """
  if config.eval_from.get('checkpoint_path') is not None:
    # uses the user-provided dir to find the model.
    ckpt_dir = config.eval_from.checkpoint_path
  assert gfile.exists(ckpt_dir)

  vq_train_state_dict = checkpoints.restore_checkpoint(ckpt_dir, None)
  vq_params = flax.core.freeze(vq_train_state_dict['ema_params'])

  vq_model = train_utils.get_vq_model(vq_config)['generator']
  parameter_overview.log_parameter_overview(vq_params)

  return vq_model, vq_params


def load_mm_vq_model(
    config: ml_collections.ConfigDict,
    rvq_quantize_steps: int = 0) -> Dict[str, TokenizerFn]:
  """Loads a pretrained multimodal vector quantization model.

  Args:
    config: Model config file.
    rvq_quantize_steps: Number of RVQ levels to perform if the RVQGAN is used.

  Returns:
    Dictionary of two functions: tokenizer and detokenizer.
  """
  vq_model, vq_params = get_mm_vq_model_and_params(config)

  if rvq_quantize_steps > 0:
    assert (
        'quantize_steps' in vq_model.config.vqvae
    ), 'quantize_steps NOT found in training.'
    vq_model.config.unlock()
    vq_model.config.vqvae.quantize_steps = rvq_quantize_steps
    vq_model.config.lock()

  enc_params, _ = vq_params.pop('decoder')
  tokenizer_fn = functools.partial(
      vq_model.apply, {'params': enc_params}, method=vq_model.encode_to_indices
  )

  dec_params, _ = vq_params.pop('encoder')
  detokenizer_fn = functools.partial(
      vq_model.apply,
      {'params': dec_params},
      method=vq_model.decode_from_indices,
  )
  # TODO(roadjiang): add audio tokenizers in this function
  return {'tokenizer': tokenizer_fn, 'detokenizer': detokenizer_fn}


def load_mm_vq_staged_decoder(
    config: ml_collections.ConfigDict, last_conv_name: str = 'Conv_4'
) -> Dict[str, TokenizerFn]:
  """Loads the two-staged VQ decoders and a standard encoder."""
  vq_model, vq_params = get_mm_vq_model_and_params(config)

  enc_params, _ = vq_params.pop('decoder')
  tokenizer_fn = functools.partial(
      vq_model.apply, {'params': enc_params}, method=vq_model.encode_to_indices
  )

  dec_params, _ = vq_params.pop('encoder')
  detokenizer_fn1 = functools.partial(
      vq_model.apply, {'params': dec_params}, method=vq_model.decode_stage1
  )

  # This is a hack for enc_dec_3dcnn or 3dcnn
  subset_params = {}
  for k, v in vq_params['decoder'].items():
    if k.startswith(last_conv_name):
      subset_params['Conv_0'] = v
      continue
  subset_params = {'decoder': subset_params}
  subset_params = flax.core.FrozenDict(subset_params)

  # parameter_overview.log_parameter_overview(new_params)
  detokenizer_fn2 = functools.partial(
      vq_model.apply, {'params': subset_params},
      method=vq_model.decode_stage2)
  return {'tokenizer': tokenizer_fn,
          'detokenizer1': detokenizer_fn1, 'detokenizer2': detokenizer_fn2}


def export_mm_vq_model_scenic(config: ml_collections.ConfigDict,
                              output_location: str,
                              batch_dim: Optional[int] = None):
  """Saves a multimodal vector quantization model as tf.SavedModel by scenic."""

  vq_model, vq_params = get_mm_vq_model_and_params(config)
  num_frames = config.get('dataset_configs.num_frames', 16)
  image_size = config.get('image_size', 256)
  latent_shape = config.get('transformer.latent_shape',
                            (num_frames // 4, image_size // 8, image_size // 8))

  enc_params, _ = vq_params.pop('decoder')
  enc_input_shape = (-1, num_frames, image_size, image_size, 3)
  def _tokenize(params, input_data):
    return vq_model.apply({'params': params},
                          input_data,
                          method=vq_model.encode_to_indices)
  export_model(
      _tokenize,
      enc_params,
      export_dir=f'{output_location}/tokenizer',
      with_gradient=False,
      batch_dim=batch_dim,
      input_shape=enc_input_shape)

  dec_params, _ = vq_params.pop('encoder')
  dec_input_shape = (-1, *latent_shape)

  def _detokenize(params, input_data):
    return vq_model.apply({'params': params},
                          input_data,
                          method=vq_model.decode_from_indices)
  export_model(
      _detokenize,
      dec_params,
      export_dir=f'{output_location}/detokenizer',
      with_gradient=False,
      input_shape=dec_input_shape,
      batch_dim=batch_dim,
      dtype=tf.int32)


def export_model(tokenizer_fn: Callable[[PyTree, PyTree], PyTree],
                 params: train_utils.PyTree,
                 export_dir: str,
                 with_gradient: bool,
                 input_shape: Tuple[int, ...] = (-1, 16, 128, 128, 3),
                 batch_dim: Optional[int] = None,
                 dtype=tf.float32):
  """Exports Flax model to Tensorflow Saved Model."""
  if batch_dim is None:
    polymorphic_shapes = '(batch, ...)'
  else:
    polymorphic_shapes = None
  input_shape = [batch_dim, *input_shape[1:]]
  input_signature = [tf.TensorSpec(input_shape, dtype)]

  logging.info('--> Exporting model to %s', export_dir)
  logging.info('--> Input shape %s', input_shape)

  export_utils.convert_and_save_model(
      tokenizer_fn,
      params,
      input_signatures=input_signature,
      polymorphic_shapes=polymorphic_shapes,
      model_dir=export_dir,
      with_gradient=with_gradient,
      enable_xla=True,
      compile_model=True)
  logging.info('Completed exporting model to %s', export_dir)


def tokenize(batch: Batch, *, tokenizer_dict: Dict[str,
                                                   TokenizerFn]) -> jnp.ndarray:
  """Tokenize video into discrete codes by the vq_model encoder."""
  return tokenizer_dict['tokenizer'](batch['inputs'])


def detokenize(codes: jnp.ndarray, *,
               tokenizer_dict: Dict[str, TokenizerFn]) -> jnp.ndarray:
  """Generate videos from the discrete codes by the vq_model decoder."""
  return tokenizer_dict['detokenizer'](codes)

