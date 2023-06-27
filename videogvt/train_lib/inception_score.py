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

"""UCF101 Inception Score evaluation.

"""

from typing import Sequence

from flax import jax_utils
from flax import linen as nn
import jax
from jax import core
from jax import lax
import jax.numpy as jnp
import numpy as np
import tensorflow_gan as tfgan
from videogvt.train_lib import metrics_lib

DEFAULT_CHECKPOINT_FILENAME = 'gs://magvit/metrics/inception_score/c3d_ucf101.flax'  # pylint: disable=line-too-long


def resize(image: jnp.ndarray, shape: core.Shape):
  """Simplified utility for resizing, matching Chainer's aligned corner resizing."""
  shape = core.canonicalize_shape(shape)
  if len(shape) != image.ndim:
    msg = ('shape must have length equal to the number of dimensions of x; '
           f' {shape} vs {image.shape}')
    raise ValueError(msg)
  if not jnp.issubdtype(image.dtype, jnp.inexact):
    image = lax.convert_element_type(image, jnp.result_type(image, jnp.float32))
  # Skip dimensions that have scale=1 and translation=0, this is only possible
  # since all of the current resize methods (kernels) are interpolating, so the
  # output = input under an identity warp.
  spatial_dims = tuple(
      i for i in range(len(shape))
      if not core.symbolic_equal_dim(image.shape[i], shape[i]))
  scale = [(core.dimension_as_value(shape[d]) - 1) /
           (core.dimension_as_value(image.shape[d]) - 1) for d in spatial_dims]
  return _scale(image, shape, spatial_dims, scale)


def _compute_weight_mat(input_size: core.DimSize, output_size: core.DimSize,
                        scale: float):
  sample_f = jnp.arange(output_size) / scale
  x = jnp.abs(sample_f[jnp.newaxis, :] -
              jnp.arange(input_size, dtype=sample_f.dtype)[:, jnp.newaxis])
  return jnp.maximum(0, 1 - jnp.abs(x))


def _scale(x, output_shape: core.Shape, spatial_dims: Sequence[int],
           scale: Sequence[float]):
  """Scales a tensor in spatial dimensions."""
  input_shape = x.shape
  assert len(input_shape) == len(output_shape)
  assert len(spatial_dims) == len(scale)
  if not spatial_dims:
    return x
  contractions = []
  in_indices = list(range(len(input_shape)))
  out_indices = list(range(len(output_shape)))
  for i, d in enumerate(spatial_dims):
    m = input_shape[d]
    n = output_shape[d]
    w = _compute_weight_mat(m, n, scale[i]).astype(x.dtype)
    contractions.append(w)
    contractions.append([d, len(output_shape) + i])
    out_indices[d] = len(output_shape) + i
  contractions.append(out_indices)
  return jnp.einsum(x, in_indices, *contractions)


class C3D(nn.Module):
  """C3D model as implemented in https://github.com/pfnet-research/tgan2."""

  @nn.compact
  def __call__(self, x):
    # input is unnormalized video, shape (B, T, H, W, C) in range [0, 1]
    if x.dtype != jnp.float32:
      raise ValueError(f'expected float32, got {x.dtype}')
    if x.ndim != 5:
      raise ValueError(
          f'expected input axes (B, T, H, W, C); received shape: {x.shape}')
    if x.shape[-1] != 3:
      raise ValueError(f'expected 3 input channels; received shape: {x.shape}')

    # original model was trained on BGR instead of RGB
    x = jnp.flip(x, axis=-1)

    # original model was trained on 112x112 frames
    x = resize(x, x.shape[:2] + (112, 112, 3))

    # original model was trained on demeaned data from [0, 255]
    x_offset = self.param('x_offset', lambda rng, s: jnp.zeros(s, jnp.float32),
                          x.shape[1:])
    x = x * 255. + x_offset[None]

    pool2d = lambda z: nn.max_pool(z, (1, 2, 2), (1, 2, 2))
    pool3d = lambda z: nn.max_pool(z, (2, 2, 2), (2, 2, 2))
    padhw = lambda z: jnp.pad(z, ((0, 0), (0, 0), (0, 1), (0, 1), (0, 0)))

    # 3d convolutional model
    x = nn.relu(nn.Conv(features=64, kernel_size=(3, 3, 3), name='conv1a')(x))
    x = pool2d(x)
    x = nn.relu(nn.Conv(features=128, kernel_size=(3, 3, 3), name='conv2a')(x))
    x = pool3d(x)
    x = nn.relu(nn.Conv(features=256, kernel_size=(3, 3, 3), name='conv3a')(x))
    x = nn.relu(nn.Conv(features=256, kernel_size=(3, 3, 3), name='conv3b')(x))
    x = pool3d(x)
    x = nn.relu(nn.Conv(features=512, kernel_size=(3, 3, 3), name='conv4a')(x))
    x = nn.relu(nn.Conv(features=512, kernel_size=(3, 3, 3), name='conv4b')(x))
    x = pool3d(x)
    x = nn.relu(nn.Conv(features=512, kernel_size=(3, 3, 3), name='conv5a')(x))
    x = nn.relu(nn.Conv(features=512, kernel_size=(3, 3, 3), name='conv5b')(x))
    x = pool3d(padhw(x))
    x = jnp.transpose(x, [0, 4, 1, 2, 3])
    x = jnp.reshape(x, x.shape[:1] + (-1,))
    x = nn.relu(nn.Dense(features=4096, name='fc6')(x))
    features = nn.relu(nn.Dense(features=4096, name='fc7')(x))
    logits = nn.Dense(features=101, name='fc8')(features)
    softmax = jax.nn.softmax(logits)
    return {'softmax': softmax, 'logits': logits, 'features': features}


def run_model(params, x):
  """Run the C3D model."""
  out = C3D().apply(params, x)
  assert out['softmax'].shape == out['logits'].shape == (x.shape[0], 101)
  return out


def inception_score_from_logits(logits, num_repeats=1):
  """Calculate the Inception Score over sampled logits."""
  logits = logits.reshape(num_repeats, -1, *logits.shape[1:])
  scores = np.array(
      [tfgan.eval.classifier_score_from_logits(l).numpy() for l in logits])
  return {'mean': scores.mean(), 'std': scores.std()}


def load_params(checkpoint_filename=None):
  if checkpoint_filename is None:
    checkpoint_filename = DEFAULT_CHECKPOINT_FILENAME
  return metrics_lib.load_params(checkpoint_filename)


def run_inception_score(params,
                        batch_iter,
                        *,
                        num_samples=10000,
                        num_repeats=4):
  """Run the C3D model with pmap on a batch iterator and get inception score."""
  params = jax_utils.replicate(params)

  def run_fn(params, x, batch_mask):
    out = run_model(params, x)
    out['batch_mask'] = batch_mask
    # Gather outputs to the first device under multinode setting
    out = jax.lax.all_gather(out, axis_name='device')
    return out

  model_fn_p = jax.pmap(run_fn, axis_name='device')
  outputs = []
  sample_count = 0
  num_samples = num_samples * num_repeats
  num_checked = 5
  for batch in batch_iter:
    if num_checked > 0:
      metrics_lib.check_input_range(batch['inputs'])
      num_checked -= 1
    out_p = model_fn_p(params, batch['inputs'], batch['batch_mask'])
    # Get flattened output from the first device
    out = jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), out_p)
    outputs.append(out)
    sample_count += out['batch_mask'].sum()
    if sample_count >= num_samples:
      break
  outputs = metrics_lib.gather_outputs_with_mask(
      outputs, num_samples=num_samples)
  scores = inception_score_from_logits(outputs['logits'], num_repeats)
  return scores
