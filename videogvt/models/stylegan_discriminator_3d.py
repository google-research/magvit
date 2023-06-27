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

"""3D StyleGAN discriminator."""

import functools
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from videogvt.models import model_utils

default_kernel_init = nn.initializers.xavier_uniform()


class ResBlock(nn.Module):
  """3D StyleGAN ResBlock for D."""
  filters: int
  norm_fn: Any
  activation_fn: Any

  @nn.compact
  def __call__(self, x):
    input_dim = x.shape[-1]
    residual = x
    x = nn.Conv(input_dim, (3, 3, 3), kernel_init=default_kernel_init)(x)
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = model_utils.downsample(x)
    residual = model_utils.downsample(residual)
    residual = nn.Conv(
        self.filters, (1, 1, 1),
        use_bias=False,
        kernel_init=default_kernel_init)(
            residual)
    x = nn.Conv(self.filters, (3, 3, 3), kernel_init=default_kernel_init)(x)
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    out = (residual + x) / math.sqrt(2)
    return out

  @nn.remat
  def remat_call(self, x):
    return self(x)


class StyleGANDiscriminator(nn.Module):
  """StyleGAN Discriminator."""

  config: ml_collections.ConfigDict
  dtype: int = jnp.float32

  def setup(self):
    self.input_size = self.config.image_size
    self.filters = self.config.discriminator.filters
    self.activation_fn = functools.partial(
        jax.nn.leaky_relu, negative_slope=0.2)
    self.channel_multipliers = self.config.discriminator.channel_multipliers
    self.norm_type = self.config.discriminator.get('norm_type')
    self.num_remat_block = self.config.discriminator.get('num_remat_blocks', 0)

  @nn.compact
  def __call__(self, x):
    norm_fn = model_utils.get_norm_layer(self.norm_type, self.dtype)
    x = nn.Conv(self.filters, (3, 3, 3), kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    filters = self.filters
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      filters = self.filters * self.channel_multipliers[i]
      if i < self.num_remat_block:
        x = ResBlock(filters, norm_fn, self.activation_fn).remat_call(x)
      else:
        x = ResBlock(filters, norm_fn, self.activation_fn)(x)
    x = nn.Conv(filters, (3, 3, 3), kernel_init=default_kernel_init)(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(filters, kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    x = nn.Dense(1, kernel_init=default_kernel_init)(x)
    return x
