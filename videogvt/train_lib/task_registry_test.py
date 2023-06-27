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

"""Tests for task_registry."""

from absl.testing import parameterized
import ml_collections
import tensorflow as tf
from videogvt.train_lib import task_registry


class TaskRegistryTest(tf.test.TestCase, parameterized.TestCase):

  def _get_config(self):
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.transformer.latent_shape = (4, 16, 16)
    config.tasks = (
        'full_generation',
        'frame_prediction',
        'frame_prediction_c',
        'frame_interpolation',
        'outpainting_h',
        'outpainting_v',
        'outpainting_c',
        'outpainting_dv',
        'inpainting_c',
        'inpainting_dv',
    )
    config.condition_mode = 'cond->input'
    config.full_generation = ml_collections.ConfigDict()
    config.full_generation.class_conditional = True
    config.frame_prediction = ml_collections.ConfigDict()
    config.frame_prediction_c = ml_collections.ConfigDict()
    config.frame_prediction_c.class_conditional = True
    config.frame_interpolation = ml_collections.ConfigDict()
    config.outpainting_h = ml_collections.ConfigDict()
    config.outpainting_h.cond_region = 'rectangle_horizontal'
    config.outpainting_v = ml_collections.ConfigDict()
    config.outpainting_v.cond_region = 'rectangle_vertical'
    config.outpainting_c = ml_collections.ConfigDict()
    config.outpainting_c.cond_region = 'rectangle_central'
    config.outpainting_dv = ml_collections.ConfigDict()
    config.outpainting_dv.cond_region = 'dynamic_vertical'
    config.outpainting_dv.cond_padding = 'constant'
    config.inpainting_c = ml_collections.ConfigDict()
    config.inpainting_c.cond_region = 'rectangle_central'
    config.inpainting_dv = ml_collections.ConfigDict()
    config.inpainting_dv.cond_region = 'dynamic_central'
    return config

  @parameterized.parameters(False, True)
  def test_tasks(self, is_train):
    config = self._get_config()
    tasks = task_registry.get_task_registry(config, is_train)
    self.assertLen(tasks, len(config.tasks))


if __name__ == '__main__':
  tf.test.main()
