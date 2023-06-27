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
r"""Configs for the VQGAN-3D on the BAIR.

"""

import ml_collections

from videogvt.configs import vqgan3d_ucf101_config

BAIR_TRAIN_SIZE = 43_264
BAIR_VAL_SIZE = 256
BAIR_TEST_SIZE = 0
VARIANT = 'VQGAN/3D'


def get_config(config_str='B'):
  """Returns the base experiment configuration."""
  version, *options = config_str.split('-')

  config = vqgan3d_ucf101_config.get_config(config_str)
  config.experiment_name = f'BAIR_{VARIANT}'

  # Overall
  config.image_size = 64
  config.num_training_epochs = {'B': 400, 'L': 800}[version]

  # Dataset.
  config.dataset_name = 'bair'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.camera_name = 'image_aux1'
  config.dataset_configs.num_classes = 0
  config.dataset_configs.frame_rate = 10
  config.dataset_configs.num_frames = 16
  config.dataset_configs.stride = 1
  config.dataset_configs.zero_centering = False  # Range is 0 to 1
  config.dataset_configs.num_eval_clips = 14  # Sample 16 frames out of 30
  config.dataset_configs.shuffle_buffer_size = 8 * config.get_ref('batch_size')
  config.dataset_configs.prefetch_to_device = 2

  # Model: vqvae
  config.vqvae.channel_multipliers = (1, 2, 4)
  # config.vqvae.custom_conv_padding = 'constant'
  config.discriminator.channel_multipliers = (2, 4, 4, 4)

  # Learning rate
  steps_per_epoch = BAIR_TRAIN_SIZE // config.get_ref('batch_size')
  config.lr_configs.steps_per_epoch = steps_per_epoch
  total_steps = config.get_ref('num_training_epochs') * steps_per_epoch
  config.lr_configs.warmup_steps = 1 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps

  config.init_from.checkpoint_path = {
      'B': 'gs://magvit/models/imagenet_2d_base_64',
      'L': 'gs://magvit/models/imagenet_2d_large_64',
  }[version]

  # Evaluation.
  config.eval.enable_inception_score = False
  config.eval.num_examples = BAIR_VAL_SIZE * 10
  config.eval.final_num_repeats = 1
  config.eval.final_num_example_multiplier = 10

  # Standalone evaluation.
  if 'eval' in options:
    config.eval_only = True
    config.eval_from.checkpoint_path = {
        'B': 'gs://magvit/models/bair_3d_base',
        'L': 'gs://magvit/models/bair_3d_large',
    }[version]
    config.eval_from.step = -1
    config.eval_from.legacy_checkpoint = True

  return config


