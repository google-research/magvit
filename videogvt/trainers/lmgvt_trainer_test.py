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

r"""Tests for maskgvt_trainer."""
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from clu import metric_writers
import jax
import tensorflow as tf
from videogvt.configs import lmgvt_ucf101_config
from videogvt.train_lib import train_utils
from videogvt.trainers import lmgvt_trainer


class TrainerTest(parameterized.TestCase):
  """Test cases for the lmgvt trainer."""

  def setUp(self):
    super().setUp()
    self.workdir = self.create_tempdir().full_path
    self.writer = metric_writers.create_default_writer(
        self.workdir, just_logging=True, asynchronous=True)

  def get_config(self):
    config = lmgvt_ucf101_config.get_config(config_str='B-runlocal')

    config.batch_size = 2
    config.eval_batch_size = config.get_ref('batch_size')
    config.debug_num_batches = 2
    config.num_training_epochs = 1

    config.multi_task = False
    config.tasks = ('full_generation',)

    del config.vq_model_from
    config.debug_pseudo_vq = True
    config.vq_codebook_size = 16

    config.transformer.latent_shape = [2, 4, 4]
    config.transformer.num_layers = 2

    config.lr_configs.steps_per_cycle = config.get_ref(
        'num_training_epochs') * config.get_ref('debug_num_batches')
    config.lr_configs.warmup_steps = 0

    config.eval_from.step = 0
    config.eval.enable_inception_score = False
    config.eval.enable_frechet_distance = False

    return config

  def test_train(self):
    config = self.get_config()
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, data_rng = jax.random.split(rng)
    dataset = train_utils.get_pseudo_dataset(config, data_rng)
    train_state, _ = lmgvt_trainer.train(
        rng=rng,
        config=config,
        dataset=dataset,
        workdir=self.workdir,
        writer=self.writer)
    del train_state
    logging.info('workdir content: %s', tf.io.gfile.listdir(self.workdir))

  def test_eval(self):
    config = self.get_config()
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, data_rng = jax.random.split(rng)
    dataset = train_utils.get_pseudo_dataset(config, data_rng)
    lmgvt_trainer.evaluate(
        rng=rng,
        config=config,
        dataset=dataset,
        workdir=self.workdir,
        writer=self.writer)
    logging.info('workdir content: %s', tf.io.gfile.listdir(self.workdir))


if __name__ == '__main__':
  absltest.main()
