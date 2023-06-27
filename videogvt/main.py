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

"""Main file for Video GVT."""

from typing import Any

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from absl import flags
from absl import logging
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from videogvt.train_lib import train_utils
from videogvt.trainers import trainer_manager
flags.DEFINE_bool('eval_while_training', False,
                  'This is a parallel evaluation job alongside training')
FLAGS = flags.FLAGS


get_trainer = trainer_manager.get_trainer
update_config_for_evaluation = lambda x: x


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the VideoGVT project."""
  rng, global_seed_rng, local_rng = jax.random.split(rng, 3)
  # Make sure each host uses a different RNG for the training data.
  is_train = not FLAGS.eval_while_training and not config.get('eval_only')

  local_rng = jax.random.fold_in(local_rng, jax.process_index())
  data_rng, local_seed_rng = jax.random.split(local_rng)
  train_utils.seed_everything(int(global_seed_rng[0]), int(local_seed_rng[0]))
  trainer = get_trainer(config.model_class, is_train=is_train)
  config = update_config_for_evaluation(config)

  dataset = train_utils.get_dataset(
      config,
      data_rng,
      is_train,
      dataset_service_address=FLAGS.dataset_service_address)
  logging.info('Trainer %s loaded', config.model_class)
  ret = trainer(
      rng=rng, config=config, dataset=dataset, workdir=workdir, writer=writer)
  # Wait until all processes are done before exiting.
  jax.pmap(jax.random.PRNGKey)(jnp.arange(
      jax.local_device_count())).block_until_ready()
  return ret


if __name__ == '__main__':
  app.run(main=main)
