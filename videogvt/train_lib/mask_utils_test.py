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

"""Tests for mask_utils."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from videogvt.train_lib import cond_utils
from videogvt.train_lib import mask_utils

BS = 4
T = 3
W = 2
H = 2
SEEDS = [0, 1, 2]


class MaskUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def _get_4d_tensor(self, bs=4, t=3, w=2, h=2):
    return np.random.uniform(0, 1024, size=(bs, t, w, h)).astype(np.int32)

  @parameterized.parameters(
      (np.random.uniform(size=(4,)), (4, 1)),
      (np.random.uniform(size=(4, 5)), (4, 5)),
      (np.random.uniform(size=(4, 5, 6)), (4, 5 * 6)),
      (np.random.uniform(size=(4, 5, 6, 7)), (4, 5 * 6 * 7)),
  )
  def test_batch_flatten(self, x, shape):
    x_flatten = mask_utils.batch_flatten(x)
    self.assertEqual(x_flatten.shape, shape)

  def test_concat_tokens(self):
    pass

  @parameterized.parameters(
      ((None, 3), (None, 4), (None, 3)),
      ((None, 4), (None, 3), (None, 3)),
      ((1, 3), (None, 4), (1, 3)),
      ((None, 3), (1, 4), (1, 3)),
      ((1, None), (2, None), (2, None)),
      ((2, None), (1, None), (2, None)),
      ((1, 3), (2, None), (2, 3)),
      ((1, None), (2, 4), (2, 4)),
      ((1, 3), (2, 4), (2, 3)),
      ((2, 3), (1, 4), (2, 3)),
      ((1, 4), (2, 3), (2, 3)),
      ((2, 4), (1, 3), (2, 3)),
  )
  def test_merge_limits(self, limit_1, limit_2, limit_gt):
    limit_new = mask_utils.merge_limits(limit_1, limit_2)
    self.assertEqual(limit_new, limit_gt)

  @parameterized.product(
      mask_ratio=[
          np.array([0.25]),
          np.array([0.48]),
          np.array([0.77]),
          np.array([1.0]),
          np.array([0.25, 0.48, 0.77, 1.0]),
      ],
      num_masked_limits=[(1, None), (1, 6), (1, 9), (1, 12), (None, 12)],
      weight_mode=[True, False],
      seed=SEEDS)
  def test_random_mask(self, mask_ratio, num_masked_limits, weight_mode,
                       seed):
    rng = jax.random.PRNGKey(seed)
    inputs = self._get_4d_tensor(*(BS, T, W, H))
    t, w, h = T, W, H
    output = mask_utils.random_mask({'inputs': inputs},
                                    rng=rng,
                                    mask_ratio=mask_ratio,
                                    weight_mode=weight_mode,
                                    num_masked_limits=num_masked_limits)

    num_total_masked = np.clip(
        np.floor((t * w * h) * mask_ratio), *
        num_masked_limits).mean() if weight_mode else t * w * h

    self.assertShapeEqual(jax.device_get(output['masked_inputs']), inputs)
    self.assertAllEqual(jax.device_get(output['targets']), inputs)
    self.assertShapeEqual(jax.device_get(output['weights']), inputs)
    self.assertEqual(
        jax.device_get(output['weights']).sum(),
        num_total_masked * inputs.shape[0])
    if weight_mode:
      self.assertAllEqual(
          jax.device_get((1.0 - output['weights']) * output['masked_inputs']),
          jax.device_get((1.0 - output['weights']) * inputs))
      self.assertAllEqual(
          jax.device_get(output['weights'] * output['masked_inputs']),
          jax.device_get(output['weights'] * mask_utils.MASK_TOKEN *
                         jnp.ones_like(inputs)))

  @parameterized.product(
      mask_ratio=[
          np.array([0.25]),
          np.array([0.25, 0.48, 0.77, 1.0]),
      ],
      num_masked_limits=[(1, None), (1, 6), (1, 9), (1, 12), (None, 12)],
      weight_mode=[True, False])
  def test_random_mask_with_or_without_block(self, mask_ratio,
                                             num_masked_limits, weight_mode):
    """Tests random_mask and random_block_mask when block_shape=(1, 1, 1)."""
    rng = jax.random.PRNGKey(0)
    inputs = self._get_4d_tensor(*(BS, T, W, H))
    output_block_mask = mask_utils.random_block_mask(
        {'inputs': inputs},
        rng=rng,
        mask_ratio=mask_ratio,
        weight_mode=weight_mode,
        num_masked_limits=num_masked_limits,
        block_shape=(1, 1, 1))
    _, subrng = jax.random.split(rng)
    output_mask = mask_utils.random_mask({'inputs': inputs},
                                         rng=subrng,
                                         mask_ratio=mask_ratio,
                                         weight_mode=weight_mode,
                                         num_masked_limits=num_masked_limits)

    self.assertAllEqual(
        jax.device_get(output_mask['masked_inputs']),
        jax.device_get(output_block_mask['masked_inputs']))
    self.assertAllEqual(
        jax.device_get(output_mask['targets']),
        jax.device_get(output_block_mask['targets']))
    self.assertAllEqual(
        jax.device_get(output_mask['weights']),
        jax.device_get(output_block_mask['weights']))

  @parameterized.product(
      mask_ratio=[
          np.array([0.25]),
          np.array([0.25, 0.48, 0.77, 1.0]),
      ],
      num_masked_limits=[(1, None)],
      weight_mode=[True],
      which_dim=['temporal', 'width', 'height'],
      seed=SEEDS,
      t=[4],
      w=[2, 6],
      h=[2, 6])
  def test_random_block_mask_corner_cases(self, mask_ratio, num_masked_limits,
                                          weight_mode, which_dim, seed, t, w,
                                          h):
    rng = jax.random.PRNGKey(seed)
    if which_dim == 'temporal':
      block_shape, dim = (t, 1, 1), 1
    elif which_dim == 'width':
      block_shape, dim = (1, w, 1), 2
    elif which_dim == 'height':
      block_shape, dim = (1, 1, h), 3
    inputs = self._get_4d_tensor(*(BS, t, w, h))
    output = mask_utils.random_block_mask({'inputs': inputs},
                                          rng=rng,
                                          mask_ratio=mask_ratio,
                                          weight_mode=weight_mode,
                                          num_masked_limits=num_masked_limits,
                                          block_shape=block_shape)

    num_total_masked = np.clip(
        np.floor((t * w * h) * mask_ratio), *
        num_masked_limits).mean() if weight_mode else t * w * h

    self.assertShapeEqual(jax.device_get(output['masked_inputs']), inputs)
    self.assertAllEqual(jax.device_get(output['targets']), inputs)
    self.assertShapeEqual(jax.device_get(output['weights']), inputs)
    self.assertEqual(
        jax.device_get(output['weights']).sum(),
        num_total_masked * inputs.shape[0])
    self.assertTrue(
        (output['masked_inputs'].mean(dim) == mask_utils.MASK_TOKEN).any())
    if weight_mode:
      self.assertAllEqual(
          jax.device_get((1.0 - output['weights']) * output['masked_inputs']),
          jax.device_get((1.0 - output['weights']) * inputs))
      self.assertAllEqual(
          jax.device_get(output['weights'] * output['masked_inputs']),
          jax.device_get(output['weights'] * mask_utils.MASK_TOKEN *
                         jnp.ones_like(inputs)))

  @parameterized.product(
      mask_ratio=[
          np.array([0.25]),
          np.array([0.25, 0.48, 0.77, 1.0]),
      ],
      num_masked_limits=[(1, None)],
      weight_mode=[True],
      seed=SEEDS,
      block_shape=[(2, 2, 1), (2, 1, 2), (2, 2, 2)])
  def test_random_block_mask_general_shape(self, mask_ratio, num_masked_limits,
                                           weight_mode, seed, block_shape):
    rng = jax.random.PRNGKey(seed)
    t, w, h = 4, 4, 4
    inputs = self._get_4d_tensor(*(BS, t, w, h))
    output = mask_utils.random_block_mask({'inputs': inputs},
                                          rng=rng,
                                          mask_ratio=mask_ratio,
                                          weight_mode=weight_mode,
                                          num_masked_limits=num_masked_limits,
                                          block_shape=block_shape)

    num_total_masked = np.clip(
        np.floor((t * w * h) * mask_ratio), *
        num_masked_limits).mean() if weight_mode else t * w * h

    self.assertShapeEqual(jax.device_get(output['masked_inputs']), inputs)
    self.assertAllEqual(jax.device_get(output['targets']), inputs)
    self.assertShapeEqual(jax.device_get(output['weights']), inputs)
    self.assertEqual(
        jax.device_get(output['weights']).sum(),
        num_total_masked * inputs.shape[0])
    if weight_mode:
      self.assertAllEqual(
          jax.device_get((1.0 - output['weights']) * output['masked_inputs']),
          jax.device_get((1.0 - output['weights']) * inputs))
      self.assertAllEqual(
          jax.device_get(output['weights'] * output['masked_inputs']),
          jax.device_get(output['weights'] * mask_utils.MASK_TOKEN *
                         jnp.ones_like(inputs)))

  # TODO(kihyuks): have the same setup for different tests.
  @parameterized.product(
      mask_ratio=[
          np.array([0.25]),
          np.array([0.25, 0.48, 0.77, 1.0]),
      ],
      cond_latent_frames=[0, 1, 2],
      condition_mode=['cond->input', 'input->input', 'cond->cond'],
      num_masked_limits=[(1, None), (1, 6), (1, 9), (1, 12), (None, 12)],
      weight_mode=[True, False],
      seed=SEEDS)
  def test_random_block_mask_with_frame_cond(self, mask_ratio,
                                             cond_latent_frames, condition_mode,
                                             num_masked_limits, weight_mode,
                                             seed):
    rng = jax.random.PRNGKey(seed)
    inputs = self._get_4d_tensor(*(BS, T, W, H))
    t, w, h = T, W, H
    block_shape = (1, 1, 1)
    cond_mask = cond_utils.frame_prediction_cond(
        jnp.zeros((BS, 2, 1, 1, 3)),
        cond_frames=1,
        cond_padding='edge',
        cond_latent_frames=cond_latent_frames,
        latent_shape=(T, W, H))['latent_mask']
    output = mask_utils.random_block_mask(
        {'inputs': inputs, 'cond_inputs': inputs, 'cond_mask': cond_mask},
        rng=rng,
        mask_ratio=mask_ratio,
        condition_mode=condition_mode,
        weight_mode=weight_mode,
        num_masked_limits=num_masked_limits,
        block_shape=block_shape)

    t_noncond = t - cond_latent_frames if condition_mode in (
        'input->input', 'cond->cond') else t
    num_total_masked = np.clip(
        np.floor((t_noncond * w * h) * mask_ratio), *
        num_masked_limits).mean() if weight_mode else t_noncond * w * h

    self.assertShapeEqual(jax.device_get(output['masked_inputs']), inputs)
    self.assertAllEqual(jax.device_get(output['targets']), inputs)
    self.assertShapeEqual(jax.device_get(output['weights']), inputs)
    self.assertEqual(
        jax.device_get(output['weights']).sum(),
        num_total_masked * inputs.shape[0])
    self.assertAllClose(
        jax.device_get(output['masked_inputs'][:, :cond_latent_frames]),
        jax.device_get(inputs[:, :cond_latent_frames]))
    if weight_mode:
      self.assertAllEqual(
          jax.device_get((1.0 - output['weights'][:, cond_latent_frames:]) *
                         output['masked_inputs'][:, cond_latent_frames:]),
          jax.device_get((1.0 - output['weights'][:, cond_latent_frames:]) *
                         inputs[:, cond_latent_frames:]))
      self.assertAllEqual(
          jax.device_get(output['weights'][:, cond_latent_frames:] *
                         output['masked_inputs'][:, cond_latent_frames:]),
          jax.device_get(output['weights'][:, cond_latent_frames:] *
                         mask_utils.MASK_TOKEN *
                         jnp.ones_like(inputs[:, cond_latent_frames:])))

  @parameterized.product(
      mask_ratio=[
          np.array([0.25]),
          np.array([0.25, 0.48, 0.77, 1.0]),
      ],
      condition_mode=['cond->input', 'input->input', 'cond->cond'],
      num_masked_limits=[(1, None), (1, 6), (1, 9), (1, 12), (None, 12)],
      weight_mode=[True, False],
      seed=SEEDS)
  def test_random_block_mask_with_or_without_cond(self, mask_ratio,
                                                  condition_mode,
                                                  num_masked_limits,
                                                  weight_mode, seed):
    rng = jax.random.PRNGKey(seed)
    inputs = self._get_4d_tensor(*(BS, T, W, H))
    block_shape = (1, 1, 1)
    output = mask_utils.random_block_mask(
        {'inputs': inputs},
        rng=rng,
        mask_ratio=mask_ratio,
        weight_mode=weight_mode,
        num_masked_limits=num_masked_limits,
        block_shape=block_shape)
    cond_mask = jnp.zeros_like(inputs, dtype=bool)
    output_with_frame_cond = mask_utils.random_block_mask(
        {'inputs': inputs, 'cond_inputs': cond_mask, 'cond_mask': cond_mask},
        rng=rng,
        mask_ratio=mask_ratio,
        condition_mode=condition_mode,
        weight_mode=weight_mode,
        num_masked_limits=num_masked_limits,
        block_shape=block_shape)

    self.assertAllEqual(
        jax.device_get(output['masked_inputs']),
        jax.device_get(output_with_frame_cond['masked_inputs']))
    self.assertAllEqual(
        jax.device_get(output['targets']),
        jax.device_get(output_with_frame_cond['targets']))
    self.assertAllEqual(
        jax.device_get(output['weights']),
        jax.device_get(output_with_frame_cond['weights']))

  @parameterized.product(
      mask_ratio=[
          np.array([0.25]),
          np.array([0.25, 0.48, 0.77, 1.0]),
      ],
      cond_latent_frames=[1, 5],
      total_frames=[9, 16],
      condition_mode=['input->input', 'cond->cond'],
      num_masked_limits=[
          (1, None),
          (1, 6),
          (1, 32),
      ],
      weight_mode=[True, False],
      seed=SEEDS)
  def test_frame_by_frame_mask(self, mask_ratio, cond_latent_frames,
                               total_frames, condition_mode, num_masked_limits,
                               weight_mode, seed):
    rng = jax.random.PRNGKey(seed)
    inputs = self._get_4d_tensor(*(BS, total_frames, W, H))
    bs, t, w, h = BS, total_frames, W, H
    block_shape = (1, 1, 1)
    cond_mask = cond_utils.frame_prediction_cond(
        jnp.zeros((BS, 2, 1, 1, 3)),
        cond_frames=1,
        cond_padding='edge',
        cond_latent_frames=cond_latent_frames,
        latent_shape=(total_frames, W, H))['latent_mask']
    output = mask_utils.frame_by_frame_mask(
        {'inputs': inputs, 'cond_inputs': inputs, 'cond_mask': cond_mask},
        rng=rng,
        mask_ratio=mask_ratio,
        condition_mode=condition_mode,
        weight_mode=weight_mode,
        num_masked_limits=num_masked_limits,
        block_shape=block_shape)

    t_noncond = t - cond_latent_frames if condition_mode in (
        'input->input', 'cond->cond') else t
    num_maskable = t_noncond * w * h
    num_masked_limits = mask_utils.merge_limits(num_masked_limits,
                                                (None, num_maskable))
    overall_mask_ratio = mask_ratio * (1 - cond_latent_frames / total_frames)
    num_total_masked = np.clip(
        np.floor((t * w * h) * overall_mask_ratio), *
        num_masked_limits).mean() if weight_mode else t_noncond * w * h

    # Manually compute unmasked frames.
    num_masked = np.clip(
        np.floor((t * w * h) * overall_mask_ratio), *num_masked_limits)
    unmasked_frames = total_frames - np.ceil(num_masked /
                                             (w * h)).astype(np.int32)
    if len(unmasked_frames) == 1:
      unmasked_frames = np.array(list(unmasked_frames) * bs)

    self.assertShapeEqual(jax.device_get(output['masked_inputs']), inputs)
    self.assertAllEqual(jax.device_get(output['targets']), inputs)
    self.assertShapeEqual(jax.device_get(output['weights']), inputs)
    self.assertEqual(
        jax.device_get(output['weights']).sum(),
        num_total_masked * inputs.shape[0])
    for b, unmasked_frame in enumerate(unmasked_frames):
      self.assertAllClose(
          jax.device_get(output['masked_inputs'][b, :unmasked_frame]),
          jax.device_get(inputs[b, :unmasked_frame]))
      self.assertAllClose(
          jax.device_get(output['masked_inputs'][b, unmasked_frame + 1:]),
          jax.device_get(mask_utils.MASK_TOKEN *
                         jnp.ones_like(inputs[b, unmasked_frame + 1:])))


if __name__ == '__main__':
  tf.test.main()
