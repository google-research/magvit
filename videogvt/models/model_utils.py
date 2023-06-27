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

"""Utility functions for Modeling."""

import functools
import math
from typing import Any, Optional, Sequence, Type

import flax.linen as nn
import jax
import jax.numpy as jnp


def get_norm_layer(norm_type, dtype):
  if norm_type == 'LN':
    norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
  elif norm_type == 'GN':
    norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
  elif norm_type is None:
    norm_fn = lambda: (lambda x: x)
  else:
    raise NotImplementedError(f'norm_type: {norm_type}')
  return norm_fn


def vmap_t_dim(module: Type[nn.Module]) -> Type[nn.Module]:
  """Vmap a 2D model to add the T dimension."""
  return nn.vmap(
      module,
      in_axes=1,
      out_axes=1,
      variable_axes={'params': None},
      split_rngs={'params': False})


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
  """Avg pooling as done by TF (Flax layer gives different results).

  To be specific, Flax includes padding cells when taking the average,
  while TF does not.

  Args:
    x: Input tensor
    window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
      2-dim tuple one gets 2d pooling.
    strides: Must have the same dimension as the window_shape.
    padding: Either 'SAME' or 'VALID' to indicate pooling method.

  Returns:
    pooled: Tensor after applying pooling.
  """
  pool_sum = jax.lax.reduce_window(
      x,
      0.0,
      jax.lax.add,
      (1,) + window_shape + (1,),
      (1,) + strides + (1,),
      padding,
  )
  pool_denom = jax.lax.reduce_window(
      jnp.ones_like(x),
      0.0,
      jax.lax.add,
      (1,) + window_shape + (1,),
      (1,) + strides + (1,),
      padding,
  )
  return pool_sum / pool_denom


def upsample_2d(x, factor=2):
  n, h, w, c = x.shape
  x = jax.image.resize(x, (n, h * factor, w * factor, c), method='nearest')
  return x


def dsample_2d(x):
  return tensorflow_style_avg_pooling(x, (2, 2), strides=(2, 2), padding='same')


def upsample(x: jnp.ndarray, include_t_dim: bool = True, factor: int = 2):
  """Upsample via nearest interpolation."""
  n, t, h, w, c = x.shape
  x = jax.image.resize(
      x, (n, t * factor if include_t_dim else t, h * factor, w * factor, c),
      method='nearest')
  return x


def downsample(x: jnp.ndarray, include_t_dim: bool = True, factor: int = 2):
  """Downsample via average pooling."""
  t_factor = factor if include_t_dim else 1
  shape = (t_factor, factor, factor)
  x = tensorflow_style_avg_pooling(x, shape, shape, padding='same')
  return x


class Conv(nn.Conv):
  """Convolution with custom padding.

  Attributes:
    custom_padding: padding mode accepted by jnp.pad. When using this, must set
      padding=VALID to disable padding in nn.Conv.
  """

  custom_padding: Optional[str] = None

  @nn.compact
  def __call__(self, x):
    if self.custom_padding is not None:
      assert self.padding == 'VALID', 'Must use VALID padding for raw Conv.'
      assert self.kernel_dilation in (1, None), 'Kernel dilation not supported.'
      pads = [((k - 1) // 2, k // 2) for k in self.kernel_size]
      pads = [(0, 0)] + pads + [(0, 0)]
      if self.custom_padding.startswith(
          'reflect_') or self.custom_padding.startswith('symmetric_'):
        custom_padding, reflect_type = self.custom_padding.split('_')
        pad_kwargs = {'reflect_type': reflect_type}
      else:
        custom_padding = self.custom_padding
        pad_kwargs = {}
      x = jnp.pad(x, pads, mode=custom_padding, **pad_kwargs)
    return super(Conv, self).__call__(x)


class Conv2Plus1D(nn.Module):
  """2+1D Separable Convolution."""

  features: int
  kernel_size: Sequence[int]
  strides: Sequence[int] = (1, 1, 1)
  padding: str = 'SAME'
  use_bias: bool = True
  dtype: Any = None
  kernel_init: Any = nn.initializers.lecun_normal()
  bias_init: Any = nn.initializers.zeros
  norm_fn: Any = None
  activation_fn: Any = nn.relu
  expand_mid_features: bool = False

  @nn.remat
  @nn.compact
  def __call__(self, x):
    assert len(self.kernel_size) == 3
    kernel_2d = (1, *self.kernel_size[1:])
    kernel_1d = (self.kernel_size[0], 1, 1)
    stride_2d = (1, *self.strides[1:])
    stride_1d = (self.strides[0], 1, 1)
    in_features = x.shape[-1]
    if self.expand_mid_features:
      mid_features = (in_features * self.features * 3 * 3 * 3) // (
          in_features * 3 * 3 + self.features * 3)
      mid_features = int(math.ceil(mid_features / 32)) * 32
      mid_features = max(self.features, mid_features)
    else:
      mid_features = self.features
    x = nn.Conv(
        mid_features,
        kernel_size=kernel_2d,
        strides=stride_2d,
        padding=self.padding,
        use_bias=self.use_bias)(
            x)
    if self.norm_fn is not None:
      x = self.norm_fn()(x)  # pylint: disable=not-callable
    x = self.activation_fn(x)
    x = nn.Conv(
        self.features,
        kernel_size=kernel_1d,
        strides=stride_1d,
        padding=self.padding,
        use_bias=self.use_bias)(
            x)
    return x


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class BlurPool2D(nn.Module):
  """A layer to do channel-wise blurring + subsampling on 2D inputs.

  Reference:
    Zhang et al. Making Convolutional Networks Shift-Invariant Again.
    https://arxiv.org/pdf/1904.11486.pdf.
  """
  filter_size: int = 4
  strides: tuple[int, int] = (2, 2)
  padding: str = 'SAME'

  def setup(self):
    if self.filter_size == 3:
      self.filter = [1., 2., 1.]
    elif self.filter_size == 4:
      self.filter = [1., 3., 3., 1.]
    elif self.filter_size == 5:
      self.filter = [1., 4., 6., 4., 1.]
    elif self.filter_size == 6:
      self.filter = [1., 5., 10., 10., 5., 1.]
    elif self.filter_size == 7:
      self.filter = [1., 6., 15., 20., 15., 6., 1.]
    else:
      raise ValueError('Only filter_size of 3, 4, 5, 6 or 7 is supported.')

    self.filter = jnp.array(self.filter, dtype=jnp.float32)
    self.filter = self.filter[:, None] * self.filter[None, :]
    with jax.default_matmul_precision('float32'):
      self.filter /= jnp.sum(self.filter)
    self.filter = jnp.reshape(
        self.filter, [self.filter.shape[0], self.filter.shape[1], 1, 1])

  @nn.compact
  def __call__(self, inputs):
    channel_num = inputs.shape[-1]
    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    depthwise_filter = jnp.tile(self.filter, [1, 1, 1, channel_num])
    with jax.default_matmul_precision('float32'):
      outputs = jax.lax.conv_general_dilated(
          inputs,
          depthwise_filter,
          self.strides,
          self.padding,
          feature_group_count=channel_num,
          dimension_numbers=dimension_numbers)
    return outputs
