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

"""Tests for *_train_lib."""

from absl.testing import parameterized
import jax
import numpy as np
import tensorflow as tf
from videogvt.train_lib import train_utils


class TrainLibTest(tf.test.TestCase, parameterized.TestCase):

  def test_long_video_iterable(self):
    # tests the normal case
    input_video = np.random.uniform(size=(1025, 16, 16, 3)).astype(np.float32)
    batch_size = 16
    temporal_window = 16
    eval_iter = iter(
        train_utils.LongVideoIterable(
            input_video, batch_size=batch_size,
            temporal_window=temporal_window))
    cnt = 0
    jax_batch_shape = (
        jax.device_count(), batch_size // jax.device_count(), temporal_window,
        *input_video.shape[1:]
    )
    for train_batch in eval_iter:
      train_batch['inputs'] = train_batch['inputs'].reshape(jax_batch_shape)
      np.testing.assert_equal(train_batch['inputs'].shape, jax_batch_shape)
      cnt += 1
    np.testing.assert_equal(
        cnt, np.floor(input_video.shape[0] / batch_size / temporal_window))

    # boundary test for short video
    input_video = np.random.uniform(size=(255, 16, 16, 3)).astype(np.float32)
    eval_iter = iter(train_utils.LongVideoIterable(input_video, 16, 16))
    cnt = 0
    for train_batch in eval_iter:
      cnt += 1
    np.testing.assert_equal(cnt, 0)

  def test_long_video_iterable_with_overlap(self):
    # tests the overlapping normal case
    input_video = np.reshape(np.arange(0, 1024), [-1, 1, 1, 1])
    batch_size = 16
    temporal_window = 16
    eval_iter = iter(
        train_utils.LongVideoIterable(
            input_video, batch_size=batch_size,
            temporal_window=temporal_window,
            overlapping_frames=0))
    l = []
    for train_batch in eval_iter:
      x = train_batch['inputs']
      x = np.squeeze(x)
      l.append(x)
      np.testing.assert_equal(x.shape[1], temporal_window)

    np.testing.assert_equal(
        np.array(l).shape, [
            np.floor(input_video.shape[0] / batch_size / temporal_window),
            batch_size, temporal_window
        ])

    # overlapping case 1
    input_video = np.reshape(np.arange(0, 16), [-1, 1, 1, 1])
    overlapping_frames = 2
    temporal_window = 3
    batch_size = 2
    eval_iter = iter(
        train_utils.LongVideoIterable(
            input_video, batch_size=batch_size,
            temporal_window=temporal_window,
            overlapping_frames=overlapping_frames))
    l = []
    for train_batch in eval_iter:
      cur_batch = np.squeeze(train_batch['inputs'])
      for i in range(1, cur_batch.shape[0]):
        np.testing.assert_array_equal(cur_batch[i - 1, -overlapping_frames:],
                                      cur_batch[i, 0:overlapping_frames])
      l.append(cur_batch)
    l = np.reshape(l, [-1])
    np.testing.assert_array_equal(l[0], 0)
    np.testing.assert_array_equal(l[-1], 15)

    # overlapping case 2
    input_video = np.reshape(np.arange(0, 200), [-1, 1, 1, 1])
    overlapping_frames = 4
    eval_iter = iter(
        train_utils.LongVideoIterable(
            input_video, batch_size=batch_size,
            temporal_window=temporal_window,
            overlapping_frames=overlapping_frames))
    l = []
    for train_batch in eval_iter:
      cur_batch = np.squeeze(train_batch['inputs'])
      for i in range(1, cur_batch.shape[0]):
        np.testing.assert_array_equal(cur_batch[i - 1, -overlapping_frames:],
                                      cur_batch[i, 0:overlapping_frames])
      l.append(cur_batch)

    # overlapping case 3
    input_video = np.random.uniform(size=(1024, 16, 16, 3)).astype(np.float32)
    overlapping_frames = 4
    temporal_window = 16
    batch_size = 8
    eval_iter = iter(
        train_utils.LongVideoIterable(
            input_video, batch_size=batch_size,
            temporal_window=temporal_window,
            overlapping_frames=overlapping_frames))
    l = []
    for train_batch in eval_iter:
      cur_batch = np.squeeze(train_batch['inputs'])
      for i in range(1, cur_batch.shape[0]):
        np.testing.assert_array_equal(cur_batch[i - 1, -overlapping_frames:],
                                      cur_batch[i, 0:overlapping_frames])
      l.append(cur_batch)
if __name__ == '__main__':
  tf.test.main()
