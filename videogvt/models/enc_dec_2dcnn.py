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

"""Encoder and Decoder stuctures modified from VQGAN paper.

https://arxiv.org/abs/2012.09841
Here we remove the non-local layer for faster speed.
"""

import functools
from typing import Any, Optional

from absl import logging
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
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    if input_dim != self.filters:
      if self.use_conv_shortcut:
        residual = self.conv_fn(
            self.filters, kernel_size=(3, 3), use_bias=False)(
                residual)
      else:
        residual = self.conv_fn(
            self.filters, kernel_size=(1, 1), use_bias=False)(
                residual)
    return x + residual


class AttnBlock(nn.Module):
  """Self-attention residual block."""

  num_heads: Optional[int]
  norm_fn: Any
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable

    assert self.num_heads is not None
    assert C % self.num_heads == 0
    num_heads = self.num_heads
    head_dim = C // num_heads

    h = self.norm_fn(name='norm')(x)

    assert h.shape == (B, H, W, C)
    h = h.reshape(B, H * W, C)
    q = nn.DenseGeneral(features=(num_heads, head_dim), name='q')(h)
    k = nn.DenseGeneral(features=(num_heads, head_dim), name='k')(h)
    v = nn.DenseGeneral(features=(num_heads, head_dim), name='v')(h)
    assert q.shape == k.shape == v.shape == (B, H * W, num_heads, head_dim)
    h = nn.dot_product_attention(query=q, key=k, value=v)
    assert h.shape == (B, H * W, num_heads, head_dim)
    h = nn.DenseGeneral(
        features=C,
        axis=(-2, -1),
        kernel_init=nn.initializers.zeros,
        name='proj_out')(h)
    assert h.shape == (B, H * W, C)
    h = h.reshape(B, H, W, C)
    assert h.shape == x.shape
    logging.info('%s: x=%r num_heads=%d head_dim=%d', self.name, x.shape,
                 num_heads, head_dim)
    return x + h


class Encoder(nn.Module):
  """Encoder Blocks."""

  config: ml_collections.ConfigDict
  dtype: int = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_enc_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.embedding_dim = self.config.vqvae.embedding_dim
    self.conv_downsample = self.config.vqvae.conv_downsample
    self.custom_conv_padding = self.config.vqvae.get('encoder_conv_padding')
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == 'relu':
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == 'swish':
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError
    # for self attn
    self.use_attn = self.config.vqvae.get('use_attn', False)
    self.num_heads = self.config.vqvae.get('num_attn_heads', 8)

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
    x = conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    num_blocks = len(self.channel_multipliers)
    filters = self.filters
    for i in range(num_blocks):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i < num_blocks - 1:
        if self.conv_downsample:
          x = conv_fn(filters, kernel_size=(4, 4), strides=(2, 2))(x)
        else:
          x = model_utils.dsample_2d(x)
    for res_idx in range(self.num_res_blocks):
      if self.use_attn and res_idx == self.num_res_blocks - 1:
        x = AttnBlock(
            num_heads=self.num_heads,
            norm_fn=norm_fn,
            dtype=self.dtype)(x)
      x = ResBlock(filters, **block_args)(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.embedding_dim, kernel_size=(1, 1))(x)
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
    self.upsample = self.config.vqvae.get('upsample', 'nearest+conv')
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == 'relu':
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == 'swish':
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError
    # for self attn
    self.use_attn = self.config.vqvae.get('use_attn', False)
    self.num_heads = self.config.vqvae.get('num_attn_heads', 8)

  @nn.compact
  def __call__(self, x, *, is_train=False, **kwargs):
    conv_fn = functools.partial(nn.Conv, dtype=self.dtype)
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
    x = conv_fn(filters, kernel_size=(3, 3), use_bias=True)(x)
    for res_idx in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
      if self.use_attn and res_idx == 0:
        x = AttnBlock(
            num_heads=self.num_heads,
            norm_fn=norm_fn,
            dtype=self.dtype)(x)
    for i in reversed(range(num_blocks)):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i > 0:
        if self.upsample == 'deconv':
          x = conv_t_fn(filters, kernel_size=(4, 4), strides=(2, 2))(x)
        elif self.upsample == 'depth_to_space':
          x = conv_fn(filters * 4, kernel_size=(3, 3))(x)
          b, h, w = x.shape[:-1]
          x = x.reshape(b, h, w, 2, 2, filters)
          x = x.transpose(0, 1, 3, 2, 4, 5)
          x = x.reshape(b, h * 2, w * 2, filters)
        elif self.upsample == 'nearest+conv':
          x = model_utils.upsample_2d(x, 2)
          x = conv_fn(filters, kernel_size=(3, 3))(x)
        else:
          raise NotImplementedError(f'Unknown upsampler: {self.upsample}')
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.output_dim, kernel_size=(3, 3))(x)
    return x
