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

# pylint: disable=line-too-long
r"""Compute FVD for the offline generated videos.

"""

import concurrent.futures
import functools
from typing import Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import numpy as np
import six
import tensorflow.compat.v1 as tf
from tensorflow.io import gfile
import tensorflow_gan as tfgan
import tensorflow_hub as hub
import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string('reference_dir', '', 'Directory to original video data')
flags.DEFINE_string('output_dir', '', 'Directory to generated video data')
flags.DEFINE_integer('num_groups', 1, 'number of groups.')

_MODULE_SPEC = 'https://tfhub.dev/deepmind/i3d-kinetics-400/1'


def normalize_videos(videos: tf.Tensor,
                     target_resolution: Tuple[int, int],
                     inter_resolution: Tuple[int, int] = (64, 64)):
  """Normalizes the videos.

  Args:
    videos: a tensor of shape [batch_size, num_frames, height, width, depth]
      ranging between 0 and 255.
    target_resolution: target video resolution of [width, height]
    inter_resolution: intermediate resolution before the target resolution.
  Returns:
    videos: a tensor of shape [batch_size, num_frames, height, width, depth]
  """
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  if tuple(videos_shape[-3:-1]) != inter_resolution:
    all_frames = tf.image.resize_bilinear(all_frames, size=inter_resolution)
  resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos


def _is_in_graph(tensor_name: str):
  """Checks whether a given tensor does exists in the graph."""
  try:
    tf.get_default_graph().get_tensor_by_name(tensor_name)
  except KeyError:
    return False
  return True


def create_i3d_embedding(videos: tf.Tensor, batch_size: int = 16):
  """Embeds the given videos.

  Args:
    videos: a tensor of shape
      [batch_size, num_frames, height=224, width=224, depth=3].
    batch_size: batch size
  Returns:
    embedding: a tensor of shape [batch_size, embedding_size].
  """

  # Making sure that we import the graph separately for
  # each different input video tensor.
  module_name = 'fvd_kinetics-400_id3_module_' + six.ensure_str(
      videos.name).replace(':', '_')
  assert_ops = [
      tf.Assert(
          tf.reduce_max(videos) <= 1.001,
          ['max value in frame is > 1', videos]),
      tf.Assert(
          tf.reduce_min(videos) >= -1.001,
          ['min value in frame is < -1', videos]),
      tf.assert_equal(
          tf.shape(videos)[0],
          batch_size, ['invalid frame batch size: ',
                       tf.shape(videos)],
          summarize=6),
  ]
  with tf.control_dependencies(assert_ops):
    videos = tf.identity(videos)

  module_scope = '%s_apply_default/' % module_name

  video_batch_size = int(videos.shape[0])
  assert video_batch_size in [batch_size, -1, None], 'Invalid batch size'
  tensor_name = module_scope + 'RGB/inception_i3d/Mean:0'
  if not _is_in_graph(tensor_name):
    i3d_model = hub.Module(_MODULE_SPEC, name=module_name)
    i3d_model(videos)

  tensor_name = module_scope + 'RGB/inception_i3d/Mean:0'
  tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
  return tensor


def caculate_fvd_from_activations(
    real_activations: np.ndarray,
    generated_activations: np.ndarray):
  """Computes the Frechet Inception distance for a batch of examples.

  Args:
    real_activations: a numpy array of shape [num_samples, embedding_size]
    generated_activations: a numpy array of shape [num_samples, embedding_size]
  Returns:
    A scalar for FVD.
  """

  return tfgan.eval.frechet_classifier_distance_from_activations(
      real_activations, generated_activations)


def get_i3d_features(data_shape: Tuple[int]):
  """Gets the i3d features for videos.

  Args:
    data_shape: data shape.
  Returns:
    A callable function to extract features.
  """
  batch_size = 256
  with tf.Graph().as_default():
    input_videos = tf.placeholder(tf.float32,
                                  shape=(batch_size,) + data_shape[1:])
    features = create_i3d_embedding(
        normalize_videos(input_videos, (224, 224)), batch_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    def feature_extractor(data, sess, input_videos, features):
      output_features = []
      num_batches = int(np.ceil(data.shape[0] / batch_size))
      for i in tqdm.tqdm(range(num_batches),
                         desc='extract features', unit='batch'):
        batch = data[i * batch_size:(i + 1) * batch_size]
        pad_length = batch_size - batch.shape[0]
        if pad_length > 0:
          batch = np.pad(
              batch, ((0, pad_length), (0, 0), (0, 0), (0, 0), (0, 0)),
              mode='constant',
              constant_values=0)
        output = sess.run(features, {input_videos: batch})
        output_features.append(output[:batch_size - pad_length])
      output_features = np.concatenate(output_features, axis=0)
      return output_features
    return functools.partial(
        feature_extractor,
        sess=sess,
        input_videos=input_videos,
        features=features)


def caculate_fvd(reference_dir: str, output_dir: str, num_groups: int):
  """compute the Frechet Vedio Distance for the generated videos.

  Args:
    reference_dir: the directory for groudtruth videos.
    output_dir: the directory for a set of generated videos.
    num_groups: the number of groups
  Returns:
    the mean and variance for the FVD value over groups
  """
  # name should be sorted as we differentiate group according to the sorted name
  #

  def read_data(video_path):
    with gfile.GFile(video_path, 'rb') as f:
      video = np.load(f)['video']
    if video.dtype == np.float32:
      assert sum((video > 1e-5) & (video < 1 - 1e-5)) == 0
    return video.astype(np.float32)

  fvds = []
  feature_fn = None
  with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
    for i in range(num_groups):
      video_features = []
      # to avoid OOM
      limit = 25600
      logging.info('processing %d-th group in %d', i, num_groups)
      for directory in [output_dir, reference_dir]:
        video_paths = [f'{directory}/{i}/{x}'
                       for x in gfile.listdir(f'{directory}/{i}')]
        features = []
        for x in range(0, len(video_paths), limit):
          videos = []
          for v in tqdm.tqdm(
              executor.map(read_data, video_paths[x:x + limit]),
              desc='load data',
              unit='example'):
            videos.append(v)
          logging.info('finish loading %d video data for %d-th group',
                       len(videos), i)
          videos = np.stack(videos)
          if feature_fn is None:
            feature_fn = get_i3d_features(videos.shape)
          features.append(feature_fn(videos))

        video_features.append(np.concatenate(features, axis=0))
        logging.info('finish extrecting %d features for %s',
                     video_features[-1].shape[0], directory)
      fvds.append(caculate_fvd_from_activations(video_features[0],
                                                video_features[1]))
  return np.mean(fvds), np.std(fvds)


def main(argv: Sequence[str]):
  del argv
  logging.info('FVD mean: %f, std: %f',
               *caculate_fvd(FLAGS.reference_dir,
                             FLAGS.output_dir, FLAGS.num_groups))

if __name__ == '__main__':
  app.run(main)
