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

"""Common functions in computing FVD/FID and Inception Score.

All functions were moved from eval_utils.py during code refactoring.
"""

from typing import Dict, List, Optional, Any

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.compat.v2.io import gfile
from videogvt.train_lib import train_utils


EvalFeatureDict = Dict[str, Any]
EvalFeatureDictCPU = Dict[str, Any]


def load_params(checkpoint_filename: str):
  with gfile.GFile(checkpoint_filename, 'rb') as fp:
    return serialization.from_bytes(None, fp.read())


def _interpolate_bilinear(im, rows, cols):
  """Bilinear interpolation, based on http://stackoverflow.com/a/12729229.

  Order of operations modified to match TF.

  Args:
    im: input image
    rows: floating point row coordinates of interpolation points
    cols: floating point column coordinates of interpolation points

  Returns:
    interpolated image
  """
  nrows, ncols = im.shape[-3:-1]

  col_lo = jnp.maximum(0, jnp.floor(cols).astype(jnp.int32))
  col_hi = jnp.minimum(col_lo + 1, ncols - 1)
  row_lo = jnp.maximum(0, jnp.floor(rows).astype(jnp.int32))
  row_hi = jnp.minimum(row_lo + 1, nrows - 1)

  im_a = im[..., row_lo, col_lo, :]
  im_b = im[..., row_hi, col_lo, :]
  im_c = im[..., row_lo, col_hi, :]
  im_d = im[..., row_hi, col_hi, :]

  # same order of operations as compute_lerp() in
  # https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/core/kernels/resize_bilinear_op.cc
  x_lerp = jnp.expand_dims(cols - col_lo, -1)
  y_lerp = jnp.expand_dims(rows - row_lo, -1)
  top = im_a + (im_c - im_a) * x_lerp
  bottom = im_b + (im_d - im_b) * x_lerp
  return top + (bottom - top) * y_lerp


def resize_bilinear(img, new_size):
  """Bilinear resizing matching the behavior of TF 1.x align_corners=False.

  See https://github.com/google/jax/issues/862#issuecomment-567674762

  Args:
    img: image
    new_size: tuple of integers representing the new height and width

  Returns:
    resized image
  """
  in_rows, in_cols = img.shape[-3:-1]
  new_rows, new_cols = new_size
  rows = jnp.arange(new_rows, dtype=jnp.float32) * (in_rows / new_rows)
  cols = jnp.arange(new_cols, dtype=jnp.float32) * (in_cols / new_cols)
  rows2, cols2 = jnp.meshgrid(rows, cols, indexing='ij')
  img_resize_vec = _interpolate_bilinear(img, rows2.flatten(), cols2.flatten())
  img_resize = img_resize_vec.reshape(
      img.shape[:-3] + (len(rows), len(cols)) + img.shape[-1:])
  return img_resize


def central_crop(img, crop_size):
  """Makes central crop of a given size.

  Args:
    img: image
    crop_size: tuple of integers representing the crop height and width

  Returns:
    cropped image
  """
  in_rows, in_cols = img.shape[-3:-1]
  crop_rows, crop_cols = crop_size
  d_rows = (in_rows - crop_rows) // 2
  d_cols = (in_cols - crop_cols) // 2
  img_crop = img[..., d_rows:d_rows + crop_rows, d_cols:d_cols + crop_cols, :]
  return img_crop


def check_input_range(x: jnp.ndarray):
  x_min, x_max = x.min(), x.max()
  if x_max > 1.001 or x_min < 0:
    raise ValueError(
        f'expected input in range [0, 1]; got range [{x_min}, {x_max}]')
  return x


def gather_outputs_with_mask(
    outputs: List[train_utils.Batch],
    *,
    mask_key: str = 'batch_mask',
    num_samples: Optional[int] = None) -> EvalFeatureDictCPU:
  """Gathers outputs using batch_mask and returns a dict."""
  outputs = jax.tree_util.tree_map(lambda *x: np.concatenate(x), *outputs)
  masks = outputs.pop(mask_key) > 0
  outputs = jax.tree_util.tree_map(lambda x: x[masks][:num_samples], outputs)
  return outputs
