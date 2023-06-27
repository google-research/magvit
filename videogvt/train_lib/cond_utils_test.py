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

"""Tests for cond_utils."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from videogvt.train_lib import cond_utils


class CondUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters('edge', 'wrap', 'constant')
  def test_frame_prediction_cond(self, cond_padding):
    cond_frames = 5
    cond_latent_frames = 1
    latent_shape = (4, 16, 16)
    input_video = np.random.uniform(size=(2, 16, 64, 64, 3)).astype(np.float32)
    input_video = jnp.asarray(input_video)
    cond_dict = cond_utils.frame_prediction_cond(
        input_video,
        cond_frames=cond_frames,
        cond_padding=cond_padding,
        cond_latent_frames=cond_latent_frames,
        latent_shape=latent_shape)
    input_video = jax.device_get(input_video)
    cond_video = jax.device_get(cond_dict['video'])
    self.assertShapeEqual(input_video, cond_video)
    self.assertAllEqual(input_video[:, :cond_frames],
                        cond_video[:, :cond_frames])
    self.assertAllEqual(
        jnp.where(cond_dict['video_mask'], cond_dict['video'], input_video),
        input_video)
    self.assertTrue(cond_dict['video_mask'][:, :cond_frames].all())
    self.assertFalse(cond_dict['video_mask'][:, cond_frames:].any())
    self.assertTrue(cond_dict['latent_mask'][:, :cond_latent_frames].all())
    self.assertFalse(cond_dict['latent_mask'][:, cond_latent_frames:].all())

  @parameterized.parameters('quarter_topleft', 'rectangle_central',
                            'dynamic_central')
  def test_inpainting_cond(self, cond_region):
    cond_padding = 'constant'
    latent_shape = (4, 16, 16)
    input_video = np.random.uniform(size=(2, 16, 64, 64, 3)).astype(np.float32)
    input_video = jnp.asarray(input_video)
    cond_dict = cond_utils.inpainting_cond(
        input_video,
        cond_region=cond_region,
        cond_padding=cond_padding,
        latent_shape=latent_shape)
    input_video = jax.device_get(input_video)
    cond_video = jax.device_get(cond_dict['video'])
    self.assertShapeEqual(input_video, cond_video)
    if cond_region == 'quarter_topleft':
      self.assertNotEqual(
          np.sum((cond_video[:, :, :32, :32, :] -
                  input_video[:, :, :32, :32, :])**2), 0.0)
      self.assertEqual(
          np.sum((cond_video[:, :, 32:64, 32:64, :] -
                  input_video[:, :, 32:64, 32:64, :])**2), 0.0)
      self.assertEqual(
          np.sum((cond_video[:, :, 32:64, :, :] -
                  input_video[:, :, 32:64, :, :])**2), 0.0)
      self.assertEqual(
          np.sum((cond_video[:, :, :, 32:64, :] -
                  input_video[:, :, :, 32:64, :])**2), 0.0)
      self.assertFalse(cond_dict['video_mask'][:, :, :32, :32, :].any())
      self.assertTrue(cond_dict['video_mask'][:, :, :32, 32:, :].all())
      self.assertTrue(cond_dict['video_mask'][:, :, 32:, :32, :].all())
      self.assertFalse(cond_dict['latent_mask'][:, :, :8, :8].any())
      self.assertTrue(cond_dict['latent_mask'][:, :, :8, 8:].all())
      self.assertTrue(cond_dict['latent_mask'][:, :, 8:, :8].all())

    elif cond_region == 'rectangle_central':
      self.assertNotEqual(
          np.sum((cond_video[:, :, 16:48, 16:48, :] -
                  input_video[:, :, 16:48, 16:48, :])**2), 0.0)
      self.assertEqual(
          np.sum(
              (cond_video[:, :, :16, :, :] - input_video[:, :, :16, :, :])**2),
          0.0)
      self.assertEqual(
          np.sum((cond_video[:, :, -16:, :, :] -
                  input_video[:, :, -16:, :, :])**2), 0.0)
      self.assertEqual(
          np.sum(
              (cond_video[:, :, :, :16, :] - input_video[:, :, :, :16, :])**2),
          0.0)
      self.assertEqual(
          np.sum((cond_video[:, :, :, -16:, :] -
                  input_video[:, :, :, -16:, :])**2), 0.0)
      self.assertFalse(cond_dict['video_mask'][:, :, 16:-16, 16:-16, :].any())
      self.assertTrue(cond_dict['video_mask'][:, :, :16, :, :].all())
      self.assertTrue(cond_dict['video_mask'][:, :, -16:, :, :].all())
      self.assertTrue(cond_dict['video_mask'][:, :, :, :16, :].all())
      self.assertTrue(cond_dict['video_mask'][:, :, :, -16:, :].all())
      self.assertFalse(cond_dict['latent_mask'][:, :, 4:-4, 4:-4].any())
      self.assertTrue(cond_dict['latent_mask'][:, :, :4, :].all())
      self.assertTrue(cond_dict['latent_mask'][:, :, -4:, :].all())
      self.assertTrue(cond_dict['latent_mask'][:, :, :, :4].all())
      self.assertTrue(cond_dict['latent_mask'][:, :, :, -4:].all())

    elif cond_region == 'dynamic_central':
      w_start = jnp.round(jnp.linspace(0, 32, 16)).astype(jnp.int32)
      w_end = w_start + 32
      for i, (w_s, w_e) in enumerate(zip(w_start, w_end)):
        self.assertNotEqual(
            np.sum((cond_video[:, i, 16:48, w_s:w_e, :] -
                    input_video[:, i, 16:48, w_s:w_e, :])**2), 0.0)
        self.assertEqual(
            np.sum((cond_video[:, i, 16:48, :w_s, :] -
                    input_video[:, i, 16:48, :w_s, :])**2), 0.0)
        self.assertEqual(
            np.sum((cond_video[:, i, 16:48, w_e:, :] -
                    input_video[:, i, 16:48, w_e:, :])**2), 0.0)
        self.assertFalse(cond_dict['video_mask'][:, i, 16:48, w_s:w_e, :].any())
        self.assertTrue(cond_dict['video_mask'][:, i, 16:48, :w_s, :].all())
        self.assertTrue(cond_dict['video_mask'][:, i, 16:48, w_e:, :].all())

      l_w_start = jnp.round(jnp.linspace(0, 8, 4)).astype(jnp.int32)
      l_w_end = l_w_start + 8
      for i, (l_s, l_e) in enumerate(zip(l_w_start, l_w_end)):
        self.assertFalse(cond_dict['latent_mask'][:, i, 4:12, l_s:l_e].any())
        self.assertTrue(cond_dict['latent_mask'][:, i, 4:12, :l_s].all())
        self.assertTrue(cond_dict['latent_mask'][:, i, 4:12, l_e:].all())


if __name__ == '__main__':
  tf.test.main()
