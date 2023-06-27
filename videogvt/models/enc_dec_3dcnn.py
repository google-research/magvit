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

"""Encoder and Decoder stuctures with 3D CNNs."""

import functools
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np
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

  @nn.remat
  def remat_call(self, x):
    return self(x)


def _get_selected_flags(total_len: int, select_len: int, suffix: bool):
  assert select_len <= total_len
  selected = np.zeros(total_len, dtype=bool)
  if not suffix:
    selected[:select_len] = True
  else:
    selected[-select_len:] = True
  return selected


class Encoder(nn.Module):
  """Encoder Blocks."""

  config: ml_collections.ConfigDict
  dtype: int = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_enc_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.temporal_downsample = self.config.vqvae.temporal_downsample
    if isinstance(self.temporal_downsample, int):
      self.temporal_downsample = _get_selected_flags(
          len(self.channel_multipliers) - 1, self.temporal_downsample, False)
    self.embedding_dim = self.config.vqvae.embedding_dim
    self.conv_downsample = self.config.vqvae.conv_downsample
    self.custom_conv_padding = self.config.vqvae.get('custom_conv_padding')
    self.norm_type = self.config.vqvae.norm_type
    self.num_remat_block = self.config.vqvae.get('num_enc_remat_blocks', 0)
    if self.config.vqvae.activation_fn == 'relu':
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == 'swish':
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x, *, is_train=False):
    conv_fn = functools.partial(
        model_utils.Conv,
        dtype=self.dtype,
        padding='VALID' if self.custom_conv_padding is not None else 'SAME',
        custom_padding=self.custom_conv_padding)
    norm_fn = model_utils.get_norm_layer(
        norm_type=self.norm_type, dtype=self.dtype)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    x = conv_fn(self.filters, kernel_size=(3, 3, 3), use_bias=False)(x)
    filters = self.filters
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        if i < self.num_remat_block and is_train:
          x = ResBlock(filters, **block_args).remat_call(x)
        else:
          x = ResBlock(filters, **block_args)(x)
      if i < num_blocks - 1:
        if self.conv_downsample:
          t_stride = 2 if self.temporal_downsample[i] else 1
          x = conv_fn(
              filters, kernel_size=(4, 4, 4), strides=(t_stride, 2, 2))(
                  x)
        else:
          x = model_utils.downsample(x, self.temporal_downsample[i])
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
    if isinstance(self.temporal_downsample, int):
      self.temporal_downsample = _get_selected_flags(
          len(self.channel_multipliers) - 1, self.temporal_downsample, False)
    self.upsample = self.config.vqvae.get('upsample', 'nearest+conv')
    self.custom_conv_padding = self.config.vqvae.get('custom_conv_padding')
    self.norm_type = self.config.vqvae.norm_type
    self.num_remat_block = self.config.vqvae.get('num_dec_remat_blocks', 0)
    if self.config.vqvae.activation_fn == 'relu':
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == 'swish':
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x, *, is_train=False, **kwargs):
    mode = kwargs.get('mode', 'all')
    conv_fn = functools.partial(
        model_utils.Conv,
        dtype=self.dtype,
        padding='VALID' if self.custom_conv_padding is not None else 'SAME',
        custom_padding=self.custom_conv_padding)
    conv_t_fn = functools.partial(nn.ConvTranspose, dtype=self.dtype)
    norm_fn = model_utils.get_norm_layer(
        norm_type=self.norm_type, dtype=self.dtype)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    num_blocks = len(self.channel_multipliers)
    filters = self.filters * self.channel_multipliers[-1]
    if mode != 'stage2':
      x = conv_fn(filters, kernel_size=(3, 3, 3), use_bias=True)(x)
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      for i in reversed(range(num_blocks)):
        filters = self.filters * self.channel_multipliers[i]
        for _ in range(self.num_res_blocks):
          if i < self.num_remat_block and is_train:
            x = ResBlock(filters, **block_args).remat_call(x)
          else:
            x = ResBlock(filters, **block_args)(x)
        if i > 0:
          t_stride = 2 if self.temporal_downsample[i - 1] else 1
          if self.upsample == 'deconv':
            assert self.custom_conv_padding is None, ('Custom padding not '
                                                      'implemented for '
                                                      'ConvTranspose')
            x = conv_t_fn(
                filters, kernel_size=(4, 4, 4), strides=(t_stride, 2, 2))(
                    x)
          elif self.upsample == 'nearest+conv':
            x = model_utils.upsample(x, self.temporal_downsample[i - 1])
            x = conv_fn(filters, kernel_size=(3, 3, 3))(x)
          else:
            raise NotImplementedError(f'Unknown upsampler: {self.upsample}')
      # self.sow('intermediates', 'preactivations', x)
      x = norm_fn()(x)
      x = self.activation_fn(x)
    if mode == 'stage1':
      return x
    x = conv_fn(self.output_dim, kernel_size=(3, 3, 3))(x)
    return x
