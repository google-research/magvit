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

"""VQVAE 3D Model."""
from typing import Any, Dict, Tuple, Type, Union, Sequence, Optional

from absl import logging
import flax.linen as nn

import jax
import jax.nn as jnn
import jax.numpy as jnp
import ml_collections
from numpy import typing as nptyping

from videogvt.models import enc_dec_2dcnn
from videogvt.models import enc_dec_2plus1dcnn
from videogvt.models import enc_dec_3dcnn
from videogvt.models import model_utils
from videogvt.train_lib import losses


ArrayLike = Union[jax.typing.ArrayLike, Sequence['ArrayLike']]
DTypeLike = nptyping.DTypeLike


def l2_normalize(x, axis=None, epsilon=1e-12):
  square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
  return jnp.multiply(x, x_inv_norm)


class VectorQuantizer(nn.Module):
  """Basic vector quantizer."""
  config: ml_collections.ConfigDict
  dtype: int = jnp.float32
  precision: Any = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self, x, *, is_train=False):
    codebook_size = self.config.vqvae.codebook_size
    codebook = self.param(
        'codebook',
        jax.nn.initializers.variance_scaling(
            scale=1.0, mode='fan_in', distribution='uniform'),
        (codebook_size, x.shape[-1]))
    codebook = jnp.asarray(codebook, dtype=self.dtype)
    if self.config.vqvae.get('latent_normalize', False):
      x = l2_normalize(x, axis=-1)
      codebook = l2_normalize(codebook, axis=-1)
    distances = jnp.reshape(
        losses.squared_euclidean_distance(
            jnp.reshape(x, (-1, x.shape[-1])), codebook),
        x.shape[:-1] + (codebook_size,))
    encoding_indices = jnp.argmin(distances, axis=-1)
    encodings = jax.nn.one_hot(
        encoding_indices, codebook_size, dtype=self.dtype)
    quantized = self.quantize(encodings)
    result_dict = dict()
    if is_train:
      e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x)**
                               2) * self.config.vqvae.commitment_cost
      q_latent_loss = jnp.mean(
          (quantized - jax.lax.stop_gradient(x))**2) * self.config.vqvae.get(
              'embedding_loss_ratio', 1.)
      entropy_loss = 0.0
      if self.config.vqvae.entropy_loss_ratio != 0:
        entropy_loss = losses.entropy_loss(
            -distances,
            loss_type=self.config.vqvae.entropy_loss_type,
            temperature=self.config.vqvae.entropy_temperature
        ) * self.config.vqvae.entropy_loss_ratio
      e_latent_loss = jnp.asarray(e_latent_loss, dtype=self.dtype)
      q_latent_loss = jnp.asarray(q_latent_loss, dtype=self.dtype)
      entropy_loss = jnp.asarray(entropy_loss, dtype=self.dtype)
      loss = e_latent_loss + q_latent_loss + entropy_loss
      result_dict = dict(
          quantizer_loss=loss,
          e_latent_loss=e_latent_loss,
          q_latent_loss=q_latent_loss,
          entropy_loss=entropy_loss)
      quantized = x + jax.lax.stop_gradient(quantized - x)

    result_dict.update({
        'encodings': encodings,
        'encoding_indices': encoding_indices,
        'raw': x,
    })
    return quantized, result_dict

  def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
    codebook = self.get_codebook()
    return jnp.dot(z, codebook, precision=self.precision)

  def get_codebook(self) -> jnp.ndarray:
    return jnp.asarray(self.variables['params']['codebook'], dtype=self.dtype)

  def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
    codebook = self.get_codebook()
    return jnp.take(codebook, ids, axis=0)




class VQVAE(nn.Module):
  """VQ-VAE model."""
  config: ml_collections.ConfigDict
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu
  precision: Any = jax.lax.Precision.DEFAULT

  def setup(self):
    """VQ-VAE setup."""
    quantizer_str = self.config.vqvae.get(
        'vector_quantizer_class', 'VectorQuantizer')
    if quantizer_str == 'VectorQuantizer':
      self.quantizer = VectorQuantizer(
          config=self.config, precision=self.precision, dtype=self.dtype
      )
    else:
      raise NotImplementedError(quantizer_str)

    if self.config.vqvae.architecture == '2dcnn':
      self.encoder = model_utils.vmap_t_dim(enc_dec_2dcnn.Encoder)(
          config=self.config, dtype=self.dtype)
      self.decoder = model_utils.vmap_t_dim(enc_dec_2dcnn.Decoder)(
          config=self.config, output_dim=3)
    elif self.config.vqvae.architecture == '3dcnn':
      self.encoder = enc_dec_3dcnn.Encoder(config=self.config, dtype=self.dtype)
      self.decoder = enc_dec_3dcnn.Decoder(config=self.config, output_dim=3)
    elif self.config.vqvae.architecture == '2plus1dcnn':
      self.encoder = enc_dec_2plus1dcnn.Encoder(
          config=self.config, dtype=self.dtype)
      self.decoder = enc_dec_2plus1dcnn.Decoder(
          config=self.config, output_dim=3)
    else:
      raise NotImplementedError(
          f'Architecture {self.config.vqvae.architecture}')

  def encode(
      self,
      x: jnp.ndarray,
      *,
      is_train: bool = False) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    video = x
    encoded_feature = self.encoder(video, is_train=is_train)
    quantized, result_dict = self.quantizer(encoded_feature, is_train=is_train)
    return quantized, result_dict  # pytype: disable=bad-return-type  # jax-ndarray

  def decode(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.decoder(x, is_train=False)

  def get_codebook_funct(self):
    # This function only works for the naive VQGAN
    return self.quantizer.get_codebook()

  def decode_from_indices(self, ids: jnp.ndarray) -> jnp.ndarray:
    features = self.quantizer.decode_ids(ids)
    reconstructed_video = self.decode(features)
    return reconstructed_video

  def decode_stage1(self, ids: jnp.ndarray) -> jnp.ndarray:
    assert self.config.vqvae.architecture == '3dcnn', 'Only support 3dcnn.'
    features = self.quantizer.decode_ids(ids)
    pre_activation_embeddings = self.decoder(
        features, is_train=False, mode='stage1')
    return pre_activation_embeddings

  def decode_stage2(self, embeddings: jnp.ndarray) -> jnp.ndarray:
    assert self.config.vqvae.architecture == '3dcnn', 'Only support 3dcnn.'
    reconstructed_video = self.decoder(
        embeddings, is_train=False, mode='stage2')
    return reconstructed_video

  def encode_to_indices(self, inputs: jnp.ndarray) -> jnp.ndarray:
    _, result_dict = self.encode(inputs, is_train=False)
    ids = result_dict['encoding_indices']
    return ids

  def __call__(
      self,
      input_video: jnp.ndarray,
      *,
      is_train: bool = False) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    quantized, result_dict = self.encode(input_video, is_train=is_train)
    outputs = self.decoder(quantized, is_train=is_train)
    return outputs, result_dict
