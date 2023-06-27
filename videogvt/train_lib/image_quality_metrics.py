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

"""Functions to compute the LPIPS, SSIM, PSNR scores.

"""
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import tensorflow as tf
import tensorflow_hub as tf_hub
from videogvt.train_lib import metrics_lib

MetricFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DEFAULT_PRECISION = jax.lax.Precision.HIGHEST


EvalFeatureDict = metrics_lib.EvalFeatureDict


def psnr_tf(im_batch1, im_batch2):
  return tf.image.psnr(im_batch1, im_batch2, max_val=1.)


def psnr(im_batch1: jnp.ndarray, im_batch2: jnp.ndarray):
  mse = jnp.mean(jnp.power((im_batch1 - im_batch2), 2), axis=(-1, -2, -3))
  return -10. * jnp.log(mse) / jnp.log(10.)


def rgb2gray(im_batch):
  return 0.2125 * im_batch[..., 0:1] + 0.7154 * im_batch[
      ..., 1:2] + 0.0721 * im_batch[..., 2:3]


def ssim_tf(im_batch1, im_batch2):
  im_batch1 = rgb2gray(im_batch1)
  im_batch2 = rgb2gray(im_batch2)
  return tf.image.ssim(im_batch1, im_batch2, max_val=1)


def ssim(im_batch1: jnp.ndarray, im_batch2: jnp.ndarray):
  im_batch1 = rgb2gray(im_batch1)
  im_batch2 = rgb2gray(im_batch2)
  return compute_ssim(im_batch1, im_batch2, max_val=1)


def compute_ssim(img0: jnp.ndarray,
                 img1: jnp.ndarray,
                 max_val: float,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 return_map: bool = False):
  """Computes SSIM from two images.


  Args:
    img0: array. An image of size [..., width, height, num_channels].
    img1: array. An image of size [..., width, height, num_channels].
    max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
    filter_size: int >= 1. Window size.
    filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
    k1: float > 0. One of the SSIM dampening parameters.
    k2: float > 0. One of the SSIM dampening parameters.
    return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

  Returns:
    Each image's mean SSIM, or a tensor of individual values if `return_map`.
  """
  # Construct a 1D Gaussian blur filter.
  hw = filter_size // 2
  shift = (2 * hw - filter_size + 1) / 2
  f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma)**2
  filt = jnp.exp(-0.5 * f_i)
  filt /= jnp.sum(filt)

  # Blur in x and y (faster than the 2D convolution).
  filt_fn1 = lambda z: jsp.signal.convolve2d(  # pylint: disable=g-long-lambda
      z, filt[:, None], mode='valid', precision=DEFAULT_PRECISION)
  filt_fn2 = lambda z: jsp.signal.convolve2d(  # pylint: disable=g-long-lambda
      z, filt[None, :], mode='valid', precision=DEFAULT_PRECISION)

  # Vmap the blurs to the tensor size, and then compose them.
  num_dims = len(img0.shape)
  map_axes = tuple(list(range(num_dims - 3)) + [num_dims - 1])
  for d in map_axes:
    filt_fn1 = jax.vmap(filt_fn1, in_axes=d, out_axes=d)
    filt_fn2 = jax.vmap(filt_fn2, in_axes=d, out_axes=d)
  filt_fn = lambda z: filt_fn1(filt_fn2(z))

  mu0 = filt_fn(img0)
  mu1 = filt_fn(img1)
  mu00 = mu0 * mu0
  mu11 = mu1 * mu1
  mu01 = mu0 * mu1
  sigma00 = filt_fn(img0**2) - mu00
  sigma11 = filt_fn(img1**2) - mu11
  sigma01 = filt_fn(img0 * img1) - mu01

  # Clip the variances and covariances to valid values.
  # Variance must be non-negative:
  sigma00 = jnp.maximum(0., sigma00)
  sigma11 = jnp.maximum(0., sigma11)
  sigma01 = jnp.sign(sigma01) * jnp.minimum(
      jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01))

  c1 = (k1 * max_val)**2
  c2 = (k2 * max_val)**2
  numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
  denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
  ssim_map = numer / denom
  ssim_score = jnp.mean(ssim_map, list(range(num_dims - 3, num_dims)))
  return ssim_map if return_map else ssim_score


def msssim_tf(im_batch1, im_batch2):
  # filter_size=11 fails. See
  # https://stackoverflow.com/questions/57127626/
  return tf.image.ssim_multiscale(im_batch1, im_batch2, max_val=1,
                                  filter_size=4)


def lpips_tf(im_batch1, im_batch2, lpips_model):
  """Not supported by tf2jax in jit or pmap."""
  return lpips_model(im_batch1, im_batch2)


def load_lpips_models():
  ret_dict = dict()
  return ret_dict


def run_models(
    video_batch1: jnp.ndarray,
    video_batch2: jnp.ndarray,
    *,
    is_tf_function: bool = True,
    metric_functions: Optional[Dict[str, MetricFunction]] = None
) -> dict[str, jnp.ndarray]:
  """Runs the jax function/models that can support jit/pmap."""
  x1, x2 = video_batch1, video_batch2
  assert x1.shape == x2.shape and x1.ndim == 5

  if metric_functions is None:
    metric_functions = dict(
        psnr=psnr,
        ssim=ssim,
    )

  flatten_t_shape = [x1.shape[0] * x1.shape[1], *x1.shape[2:]]
  flatten_x1 = jnp.reshape(x1, flatten_t_shape)
  flatten_x2 = jnp.reshape(x2, flatten_t_shape)

  if is_tf_function:
    flatten_x1 = jax.tree_map(tf.convert_to_tensor, flatten_x1)
    flatten_x2 = jax.tree_map(tf.convert_to_tensor, flatten_x2)

  ret_dict = dict()
  for (k, func) in metric_functions.items():
    flatten_metric_scores = func(flatten_x1, flatten_x2)
    if is_tf_function:
      flatten_metric_scores = flatten_metric_scores._numpy()  # pylint: disable=protected-access  # pytype: disable=attribute-error  # jax-ndarray

    ret_dict[k] = jnp.mean(
        jnp.reshape(flatten_metric_scores, x1.shape[0:2]), -1)
  return ret_dict
