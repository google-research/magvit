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
r"""Configuration and hyperparameter for the MaskGVT on BAIR multi-task.

"""

import ml_collections

from videogvt.configs import maskgvt_bair_config


VARIANT = 'MaskGVT/16'


def get_config(config_str='B'):
  """Get the base hyperparameter configuration."""
  assert config_str in ['runlocal', 'eval', 'B', 'L'], f'Unknown config_str: {config_str}'
  eval_job = False
  runlocal = False
  if config_str == 'runlocal':
    runlocal = True
    version = 'B'
  elif config_str == 'eval':
    eval_job = True
    version = 'B'
  else:
    version = config_str

  config = maskgvt_bair_config.get_config('runlocal' if runlocal else version)
  config.experiment_name = f'BAIR_MT_{VARIANT}'

  # Overall
  config.num_training_epochs = {'B': 1200, 'L': 1600}[version]

  # Model: MaskGVT
  config.multi_task = True
  config.tasks = (
      'frame_prediction',
      'frame_interpolation',
      'outpainting_h',
      'outpainting_v',
      'outpainting_c',
      'outpainting_dv',
      'inpainting_c',
      'inpainting_dv',
  )
  config.weight_mode = 'mask+refine+recons'
  config.condition_mode = 'cond->input'
  config.frame_prediction = ml_collections.ConfigDict()
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

  # Evaluation
  config.eval.data_splits = 'train,validation'
  config.eval.final_num_repeats = 1
  config.sampling.replace_condition_pixels = False

  # Standalone evaluation.
  if eval_job:
    config.eval_only = True
    config.eval.task = 'frame_prediction'
    config.eval.data_splits = 'validation'
    config.eval_from.xm = (-1, 1)
    config.eval_from.step = -1
    config.eval.results_dir = None

  return config


