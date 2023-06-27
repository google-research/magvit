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

"""Tests for frechet_distance."""

import jax
import numpy as np
import tensorflow as tf
from videogvt.train_lib import frechet_distance


class FrechetDistanceTest(tf.test.TestCase):

  def get_batch_iter(self):
    while True:
      batch = {
          'inputs':  # (device, local_batch, T, H, W, C)
              np.random.uniform(size=(1, 1, 16, 64, 64, 3)).astype(np.float32),
          'batch_mask':  # (device, local_batch)
              np.ones((1, 1), dtype=np.float32),
      }
      yield batch

  def test_run_frechet_distance(self):
    rng = jax.random.PRNGKey(0)
    batch_iter = self.get_batch_iter()
    x = next(batch_iter)['inputs'][0]
    params = frechet_distance.I3D().init(rng, x)
    scores = frechet_distance.run_frechet_distance(
        params, batch_iter, batch_iter, num_samples=2, num_repeats=2)
    self.assertContainsSubset(['mean', 'std'], scores.keys())

if __name__ == '__main__':
  tf.test.main()
