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

"""Frechet Video Distance evaluation.

"""

from typing import Tuple

from flax import jax_utils
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_gan as tfgan
from videogvt.train_lib import metrics_lib

# High precision convs and matmuls to ensure that the network's behavior on
# TPU is accurate compared to CPU/GPU.
_PRECISION = 'float32'
DEFAULT_CHECKPOINT_FILENAME = 'gs://magvit/metrics/frechet_distance/i3d_k400.flax'  # pylint: disable=line-too-long


class FrozenBatchNorm(nn.Module):
  """I3D frozen batch normalization."""

  epsilon: float = 0.001

  @nn.compact
  def __call__(self, x):
    dim = x.shape[-1]
    # this network has no gamma
    # gamma = self.param('gamma', nn.initializers.ones, (dim,))
    beta = self.param('beta', nn.initializers.zeros, (dim,))
    mean = self.param('moving_mean', nn.initializers.zeros, (dim,))
    var = self.param('moving_variance', nn.initializers.ones, (dim,))
    return (x - mean) * jax.lax.rsqrt(var + self.epsilon) + beta


class Unit3D(nn.Module):
  """I3D basic block."""

  output_channels: int
  kernel_shape: Tuple[int, int, int] = (1, 1, 1)
  stride: Tuple[int, int, int] = (1, 1, 1)
  use_batch_norm: bool = True
  use_bias: bool = False
  use_relu: bool = True

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
        features=self.output_channels,
        kernel_size=self.kernel_shape,
        strides=self.stride,
        use_bias=self.use_bias,
        precision=_PRECISION,
        name='conv_3d')(
            x)
    if self.use_batch_norm:
      x = FrozenBatchNorm(name='batch_norm')(x)
    return jax.nn.relu(x) if self.use_relu else x


class Branch(nn.Module):
  """I3D branch."""

  chns: Tuple[int, ...]  # (64, 96, 128, 16, 32, 32)

  @nn.compact
  def __call__(self, x):
    assert len(self.chns) == 6
    b0 = Unit3D(self.chns[0], name='Branch_0.Conv3d_0a_1x1')(x)
    b1 = Unit3D(self.chns[1], name='Branch_1.Conv3d_0a_1x1')(x)
    b1 = Unit3D(self.chns[2], (3, 3, 3), name='Branch_1.Conv3d_0b_3x3')(b1)
    b2 = Unit3D(self.chns[3], name='Branch_2.Conv3d_0a_1x1')(x)
    b2 = Unit3D(self.chns[4], (3, 3, 3), name='Branch_2.Conv3d_0b_3x3')(b2)
    b3 = nn.max_pool(x, (3, 3, 3), strides=(1, 1, 1), padding='SAME')
    b3 = Unit3D(self.chns[5], name='Branch_3.Conv3d_0b_1x1')(b3)
    return jnp.concatenate([b0, b1, b2, b3], axis=4)


class I3D(nn.Module):
  """I3D network."""

  num_classes: int = 400

  @nn.compact
  def __call__(self, x, *, resize_crop: bool = False):
    # input is unnormalized video, shape (B, T, H, W, C) in range [0, 1]
    if x.dtype != jnp.float32:
      raise ValueError(f'expected float32, got {x.dtype}')
    if x.ndim != 5:
      raise ValueError(
          f'expected input axes (B, T, H, W, C); received shape: {x.shape}')
    if x.shape[1] < 9:
      raise ValueError(
          f'the I3D network expects >= 9 frames; received shape: {x.shape}')

    out = {}

    # resize to (224, 224) and scale to [-1, 1]
    if resize_crop:
      x = metrics_lib.resize_bilinear(x,
                                      (256, 256 * x.shape[-2] // x.shape[-3]))
      x = metrics_lib.central_crop(x, (224, 224))
    else:
      x = out['scaled'] = metrics_lib.resize_bilinear(x, (224, 224))
    x = out['scaled'] = x * 2 - 1

    # stem
    x = out['Conv3d_1a_7x7'] = Unit3D(
        64, (7, 7, 7), (2, 2, 2), name='Conv3d_1a_7x7')(
            x)
    x = out['MaxPool3d_2a_3x3'] = nn.max_pool(
        x, (1, 3, 3), strides=(1, 2, 2), padding='SAME')
    x = out['Conv3d_2b_1x1'] = Unit3D(64, name='Conv3d_2b_1x1')(x)
    x = out['Conv3d_2c_3x3'] = Unit3D(192, (3, 3, 3), name='Conv3d_2c_3x3')(x)
    x = out['MaxPool3d_3a_3x3'] = nn.max_pool(
        x, (1, 3, 3), strides=(1, 2, 2), padding='SAME')

    # branches
    x = out['Mixed_3b'] = Branch((64, 96, 128, 16, 32, 32), name='Mixed_3b')(x)
    x = out['Mixed_3c'] = Branch((128, 128, 192, 32, 96, 64), name='Mixed_3c')(
        x)
    x = out['MaxPool3d_4a_3x3'] = nn.max_pool(
        x, (3, 3, 3), strides=(2, 2, 2), padding='SAME')
    x = out['Mixed_4b'] = Branch((192, 96, 208, 16, 48, 64), name='Mixed_4b')(x)
    x = out['Mixed_4c'] = Branch((160, 112, 224, 24, 64, 64), name='Mixed_4c')(
        x)
    x = out['Mixed_4d'] = Branch((128, 128, 256, 24, 64, 64), name='Mixed_4d')(
        x)
    x = out['Mixed_4e'] = Branch((112, 144, 288, 32, 64, 64), name='Mixed_4e')(
        x)
    x = out['Mixed_4f'] = Branch((256, 160, 320, 32, 128, 128),
                                 name='Mixed_4f')(
                                     x)
    x = out['MaxPool3d_5a_2x2'] = nn.max_pool(
        x, (2, 2, 2), strides=(2, 2, 2), padding='SAME')
    x = out['Mixed_5b'] = Branch((256, 160, 320, 32, 128, 128),
                                 name='Mixed_5b')(
                                     x)
    x = out['Mixed_5c'] = Branch((384, 192, 384, 48, 128, 128),
                                 name='Mixed_5c')(
                                     x)

    # classification
    x = out['AvgPool3d_6a_7x7'] = nn.avg_pool(
        x, (2, 7, 7), strides=(1, 1, 1), padding='VALID')
    logits = Unit3D(
        self.num_classes, (1, 1, 1),
        use_relu=False,
        use_batch_norm=False,
        use_bias=True,
        name='Logits.Conv3d_0c_1x1')(
            x)
    logits = out['Logits'] = logits.squeeze(axis=(2, 3))
    # the standard FVD implementation uses logits_mean for features
    logits_mean = out['LogitsMean'] = logits.mean(axis=1)
    out['Predictions'] = jax.nn.softmax(logits_mean)
    return out


def run_model(params, x, **model_args):
  """Run the I3D model."""
  out = I3D().apply(params, x, **model_args)
  out = {
      'softmax': out['Predictions'],
      'logits_mean': out['LogitsMean'],
      'pool': out['AvgPool3d_6a_7x7'].squeeze(axis=(2, 3)).mean(axis=1),
  }
  assert out['softmax'].shape == out['logits_mean'].shape == (x.shape[0], 400)
  assert out['pool'].shape == (x.shape[0], 1024)
  return out


def frechet_distance_from_logits(logits_1, logits_2, num_repeats=1):
  logits_1 = logits_1.reshape(num_repeats, -1, *logits_1.shape[1:])
  logits_2 = logits_2.reshape(num_repeats, -1, *logits_2.shape[1:])
  scores = np.array([
      tfgan.eval.frechet_classifier_distance_from_activations(l1, l2).numpy()
      for l1, l2 in zip(logits_1, logits_2)
  ])
  return {'mean': scores.mean(), 'std': scores.std()}


def load_params(checkpoint_filename=None):
  if checkpoint_filename is None:
    checkpoint_filename = DEFAULT_CHECKPOINT_FILENAME
  return metrics_lib.load_params(checkpoint_filename)


def run_frechet_embeddings(params,
                           batch_iter,
                           *,
                           num_samples=10000,
                           **model_args):
  """Run the I3D model with pmap on a batch iterator."""
  params = jax_utils.replicate(params)

  def run_fn(params, x, batch_mask):
    out = run_model(params, x, **model_args)
    out['batch_mask'] = batch_mask
    # Gather outputs to the first device under multinode setting
    out = jax.lax.all_gather(out, axis_name='device')
    return out

  model_fn_p = jax.pmap(run_fn, axis_name='device')
  outputs = []
  sample_count = 0
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
  return outputs


def run_frechet_distance(params,
                         batch_iter_1,
                         batch_iter_2,
                         *,
                         num_samples=10000,
                         num_repeats=4,
                         **model_args):
  """Run the I3D model with pmap on two batch iterators and get distance."""
  embeddings_1 = run_frechet_embeddings(
      params, batch_iter_1, num_samples=num_samples * num_repeats,
      **model_args)
  embeddings_2 = run_frechet_embeddings(
      params, batch_iter_2, num_samples=num_samples * num_repeats,
      **model_args)
  scores = frechet_distance_from_logits(embeddings_1['logits_mean'],
                                        embeddings_2['logits_mean'],
                                        num_repeats)
  return scores
