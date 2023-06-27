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

"""Trainer Manager."""
from typing import Any

from absl import logging
from videogvt.trainers import lmgvt_trainer
from videogvt.trainers import maskgvt_trainer
from videogvt.trainers import vqgan_trainer


TRAINER_INFO = {
    'VQGAN': {
        'class': vqgan_trainer,
        'train': vqgan_trainer.train,
        'eval': vqgan_trainer.evaluate,
    },
    'MaskGVT': {
        'class': maskgvt_trainer,
        'train': maskgvt_trainer.train,
        'eval': maskgvt_trainer.evaluate,
    },
    'LMGVT': {
        'class': lmgvt_trainer,
        'train': lmgvt_trainer.train,
        'eval': lmgvt_trainer.evaluate,
    },
}


def get_trainer_name(input_trainer: Any):
  for (k, trainer_info) in TRAINER_INFO.items():
    if input_trainer == trainer_info['class']:
      return k
  raise ValueError(f'Unkown trainer: {input_trainer}.')


def get_trainer_class(model_class: str):
  assert model_class in TRAINER_INFO
  return TRAINER_INFO[model_class]['class']


def get_trainer(model_class: str, is_train: bool):
  """Returns trainer given its name."""
  logging.info('Running model_class %s in is_train %s.', model_class, is_train)
  try:
    if is_train:
      return TRAINER_INFO[model_class]['train']
    else:
      return TRAINER_INFO[model_class]['eval']
  except KeyError as key_error:
    raise ValueError(f'Unsupported trainer: {model_class}.') from key_error
