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

"""Encoder and Decoder stuctures with 2+1D CNNs."""

import functools
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from videogvt.models import model_utils


class ResBlock(nn.Module):
  """Basic Residual Block."""
  filters: int
  norm_fn: Any
  conv_fn: Any
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu
  use_conv_shortcut: bool = False

  @nn.compact
  def __call__(self, x):
    input_dim = x.shape[-1]
    residual = x
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3, 3), use_bias=False)(x)
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3, 3), use_bias=False)(x)
    if input_dim != self.filters:
      if self.use_conv_shortcut:
        residual = self.conv_fn(
            self.filters, kernel_size=(3, 3, 3), use_bias=False)(
                residual)
      else:
        residual = self.conv_fn(
            self.filters, kernel_size=(1, 1, 1), use_bias=False)(
                residual)
    return x + residual


class Encoder(nn.Module):
  """Encoder Blocks."""

  config: ml_collections.ConfigDict
  dtype: int = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_enc_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.temporal_downsample = self.config.vqvae.temporal_downsample
    self.embedding_dim = self.config.vqvae.embedding_dim
    self.conv_downsample = self.config.vqvae.conv_downsample
    self.expand_mid_features = self.config.vqvae.expand_mid_features
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == 'relu':
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == 'swish':
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x, *, is_train=False):
    norm_fn = model_utils.get_norm_layer(
        norm_type=self.norm_type, dtype=self.dtype)
    conv_fn = functools.partial(
        model_utils.Conv2Plus1D,
        dtype=self.dtype,
        norm_fn=norm_fn,
        activation_fn=self.activation_fn,
        expand_mid_features=self.expand_mid_features)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    x = conv_fn(self.filters, kernel_size=(3, 3, 3), use_bias=False)(x)
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i < num_blocks - 1:
        if self.conv_downsample:
          t_stride = 2 if self.temporal_downsample[i - 1] else 1
          x = conv_fn(
              filters, kernel_size=(4, 4, 4), strides=(t_stride, 2, 2))(
                  x)
        else:
          x = model_utils.downsample(x, self.temporal_downsample[i - 1])
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.embedding_dim, kernel_size=(1, 1, 1))(x)
    return x


class Decoder(nn.Module):
  """Decoder Blocks."""

  config: ml_collections.ConfigDict
  output_dim: int = 3
  dtype: Any = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_dec_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.temporal_downsample = self.config.vqvae.temporal_downsample
    self.deconv_upsample = self.config.vqvae.deconv_upsample
    self.expand_mid_features = self.config.vqvae.expand_mid_features
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == 'relu':
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == 'swish':
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x, *, is_train=False, **kwargs):
    conv_t_fn = nn.ConvTranspose
    norm_fn = model_utils.get_norm_layer(
        norm_type=self.norm_type, dtype=self.dtype)
    conv_fn = functools.partial(
        model_utils.Conv2Plus1D,
        dtype=self.dtype,
        norm_fn=norm_fn,
        activation_fn=self.activation_fn,
        expand_mid_features=self.expand_mid_features)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    num_blocks = len(self.channel_multipliers)
    filters = self.filters * self.channel_multipliers[-1]
    x = conv_fn(filters, kernel_size=(3, 3, 3), use_bias=True)(x)
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    for i in reversed(range(num_blocks)):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i > 0:
        if self.deconv_upsample:
          t_stride = 2 if self.temporal_downsample[i - 1] else 1
          x = conv_t_fn(
              filters, kernel_size=(4, 4, 4), strides=(t_stride, 2, 2))(
                  x)
          raise NotImplementedError
        else:
          x = model_utils.upsample(x, self.temporal_downsample[i - 1])
          x = conv_fn(filters, kernel_size=(3, 3, 3))(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.output_dim, kernel_size=(3, 3, 3), norm_fn=None)(x)
    return x
