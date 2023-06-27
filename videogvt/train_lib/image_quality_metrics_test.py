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

"""Tests for image_quality_metrics."""

import functools

import jax
import numpy as np
import tensorflow as tf

from videogvt.train_lib import image_quality_metrics


class ImageQualityMetricsTest(tf.test.TestCase):

  def get_batch_iter(self, num_device=4, bs=2):
    while True:
      batch = {
          'inputs':  # (device, local_batch, T, H, W, C)
              np.random.uniform(size=(num_device, bs, 16, 64, 64,
                                      3)).astype(np.float32),
          'generated':
              np.random.uniform(size=(num_device, bs, 16, 64, 64, 3)
                               ).astype(np.float32),
          'batch_mask':
              np.ones((num_device, bs), dtype=np.float32),
      }
      yield batch

  def test_metric_function(self):
    bs = 8
    batch_iter = self.get_batch_iter(bs=bs)
    x = jax.tree_util.tree_map(lambda x: x[0], next(batch_iter))

    # psnr
    gt = image_quality_metrics.psnr_tf(tf.convert_to_tensor(x['inputs']),
                                       tf.convert_to_tensor(x['generated']))
    out1 = image_quality_metrics.psnr(x['generated'], x['inputs'])
    np.testing.assert_array_almost_equal(gt, out1, decimal=3)

    # ssim
    gt = image_quality_metrics.ssim_tf(tf.convert_to_tensor(x['inputs']),
                                       tf.convert_to_tensor(x['generated']))
    out2 = image_quality_metrics.ssim(x['generated'], x['inputs'])
    np.testing.assert_array_almost_equal(gt, out2)

  def test_jax_function_pmapped(self):
    # run_models for jax functions.
    bs = 8
    batch_iter = self.get_batch_iter(num_device=1, bs=bs)
    next(batch_iter)
    x = next(batch_iter)

    run_model_pmapped = jax.pmap(
        functools.partial(
            image_quality_metrics.run_models,
            is_tf_function=False,
        ),
    )
    out = run_model_pmapped(x['generated'], x['inputs'])
    for i in range(x['inputs'].shape[0]):
      gt_psnr = image_quality_metrics.psnr(x['inputs'][i], x['generated'][i])
      np.testing.assert_array_almost_equal(
          np.mean(gt_psnr, -1), out['psnr'][i])
      gt_ssim = image_quality_metrics.ssim(x['inputs'][i], x['generated'][i])
      np.testing.assert_array_almost_equal(
          np.mean(gt_ssim, -1), out['ssim'][i])

if __name__ == '__main__':
  tf.test.main()
