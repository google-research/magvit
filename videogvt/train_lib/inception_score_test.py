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

"""Tests for inception_score_ucf."""

import jax
import numpy as np
import tensorflow as tf
from videogvt.train_lib import inception_score


class InceptionScoreTest(tf.test.TestCase):

  def get_batch_iter(self):
    while True:
      batch = {
          'inputs':  # (device, local_batch, T, H, W, C)
              np.random.uniform(size=(1, 1, 16, 64, 64, 3)).astype(np.float32),
          'batch_mask':  # (device, local_batch)
              np.ones((1, 1), dtype=np.float32),
      }
      yield batch

  def test_run_inception_score(self):
    rng = jax.random.PRNGKey(0)
    batch_iter = self.get_batch_iter()
    x = next(batch_iter)['inputs'][0]
    params = inception_score.C3D().init(rng, x)
    scores = inception_score.run_inception_score(
        params, batch_iter, num_samples=2, num_repeats=2)
    self.assertContainsSubset(['mean', 'std'], scores.keys())


if __name__ == '__main__':
  tf.test.main()
