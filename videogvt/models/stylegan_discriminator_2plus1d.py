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
from flax.linen.initializers import xavier_uniform
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
from videogvt.models import model_utils

default_kernel_init = xavier_uniform()


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class ResBlock(nn.Module):
  """2+1D StyleGAN ResBlock for D."""
  filters: int
  conv_fn: Any
  activation_fn: Any

  @nn.compact
  def __call__(self, x):
    input_dim = x.shape[-1]
    residual = x
    x = self.conv_fn(input_dim, (3, 3, 3), kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    x = model_utils.downsample(x)
    residual = model_utils.downsample(residual)
    residual = nn.Conv(
        self.filters, (1, 1, 1),
        use_bias=False,
        kernel_init=default_kernel_init)(
            residual)
    x = self.conv_fn(
        self.filters, (3, 3, 3), kernel_init=default_kernel_init)(
            x)
    x = self.activation_fn(x)
    out = (residual + x) / math.sqrt(2)
    return out


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
    self.expand_mid_features = self.config.discriminator.expand_mid_features

  @nn.compact
  def __call__(self, x):
    conv_fn = functools.partial(
        model_utils.Conv2Plus1D,
        activation_fn=self.activation_fn,
        expand_mid_features=self.expand_mid_features)
    x = conv_fn(self.filters, (3, 3, 3), kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      filters = self.filters * self.channel_multipliers[i]
      x = ResBlock(filters, conv_fn, self.activation_fn)(x)
    x = conv_fn(filters, (3, 3, 3), kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(filters, kernel_init=default_kernel_init)(x)
    x = self.activation_fn(x)
    x = nn.Dense(1, kernel_init=default_kernel_init)(x)
    return x
