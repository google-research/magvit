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
r"""Configuration and hyperparameter for the MaskGVT on UCF101.

"""

import ml_collections


UCF101_TRAIN_SIZE = 9_537
UCF101_TEST_SIZE = 3_783
NUM_CLASSES = 101
VARIANT = 'MaskGVT/16'


def get_config(config_str='B'):
  """Get the base hyperparameter configuration."""
  version, *options = config_str.split('-')

  config = ml_collections.ConfigDict()
  config.experiment_name = f'UCF101_CG_{VARIANT}'

  # Overall
  config.rng_seed = 0
  config.image_size = 128
  config.batch_size = 256
  config.eval_batch_size = config.get_ref('batch_size') // 4
  config.num_training_epochs = 2000
  config.lax_precision = 'default'

  # Dataset.
  config.dataset_name = 'ucf101'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.num_classes = NUM_CLASSES

  config.data_dtype_str = 'float32'
  config.dataset_configs.num_frames = 16
  config.dataset_configs.stride = 1
  config.dataset_configs.min_resize = config.get_oneway_ref('image_size')
  config.dataset_configs.crop_size = config.get_oneway_ref('image_size')
  config.dataset_configs.one_hot_label = False
  config.dataset_configs.zero_centering = False    # Range is 0 to 1
  config.dataset_configs.num_test_clips = 1
  config.dataset_configs.prefetch_to_device = 2

  # Model: MaskGVT
  model_class, _ = VARIANT.split('/')
  config.model_class = model_class
  config.dtype = config.get_ref('data_dtype_str')
  config.tasks = ('full_generation',)
  config.full_generation = ml_collections.ConfigDict()
  config.full_generation.class_conditional = True

  config.label_smoothing = 1e-4
  config.mask_scheduling_method = 'cosine'
  config.total_seq_length = -1  # placeholder

  # VQ Model
  config.vq_model_from = ml_collections.ConfigDict()
  from videogvt.configs import vqgan3d_ucf101_config
  config.vq_model_from.config = vqgan3d_ucf101_config.get_config(f'{version}-eval')
  config.vq_codebook_size = -1  # placeholder

  # Transformer model
  config.transformer = ml_collections.ConfigDict()
  config.transformer.latent_shape = [4] + [16] * 2  # [l_t, l_h, l_w]
  config.transformer.num_layers = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24, 'H': 32}[version]
  config.transformer.hidden_size = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024, 'H': 1280}[version]
  config.transformer.mlp_dim = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096, 'H': 5120}[version]
  config.transformer.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
  config.transformer.dropout_rate = 0.1
  config.transformer.attention_dropout_rate = 0.1  # [0.0]
  config.transformer.stochastic_depth = 0.0  # TODO(roadjiang): support this

  # Learning rate
  config.base_lr = 1.0e-4
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.lr = config.get_ref('base_lr')
  config.optimizer.beta1 = 0.9
  config.optimizer.beta2 = 0.96  # [0.999, 0.98, 0.99]
  config.optimizer.weight_decay = 4.5e-2  # 0
  config.max_grad_norm = 1.0

  steps_per_epoch = UCF101_TRAIN_SIZE // config.get_ref('batch_size')
  total_steps = config.get_ref('num_training_epochs') * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 5000
  config.lr_configs.steps_per_cycle = total_steps
  # config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.base_learning_rate = config.get_ref('base_lr')

  # Sampling during inference.
  config.sampling = ml_collections.ConfigDict()
  config.sampling.mask_bins = 12
  config.sampling.choice_temperature = 4.5
  config.sampling.mask_scheduling_method = config.get_oneway_ref(
      'mask_scheduling_method')

  # Evaluation.
  config.eval = ml_collections.ConfigDict()
  config.eval.enable_inception_score = True
  config.eval.enable_frechet_distance = True
  config.eval.data_splits = 'train'
  config.eval.num_examples = 10000
  config.eval.final_num_repeats = 4

  config.eval_from = ml_collections.ConfigDict()
  config.eval_from.checkpoint_path = None
  config.eval_from.step = None

  # Logging.
  config.logging = ml_collections.ConfigDict()
  config.logging.enable_checkpoint = True
  config.logging.checkpoint_steps = 1000
  config.logging.checkpoint_kept = 5
  config.logging.log_metric_steps = 500
  config.logging.log_sample_size = 8

  if 'runlocal' in options:
    config.batch_size = 16
    config.num_training_epochs = 1
    config.transformer.num_layers = 6
    config.transformer.hidden_size = 64
    config.transformer.mlp_dim = 128
    config.transformer.num_heads = 2
    config.logging.enable_checkpoint = False
    config.logging.checkpoint_steps = 100
    config.logging.log_metric_steps = 20

  # Standalone evaluation.
  if 'eval' in options:
    config.eval_only = True
    config.eval_from.checkpoint_path = {
        'B': 'gs://magvit/models/ucf_gvt_base_cg',
        'L': 'gs://magvit/models/ucf_gvt_large_cg',
    }[version]
    config.eval_from.step = -1

  return config


