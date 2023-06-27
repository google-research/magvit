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

r"""Tests for vqgan_trainer."""
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from clu import metric_writers
import jax
import tensorflow as tf

from videogvt.configs import vqgan2d_ucf101_config
from videogvt.configs import vqgan3d_ucf101_config
from videogvt.train_lib import train_utils
from videogvt.trainers import vqgan_trainer


class TrainerTest(parameterized.TestCase):
  """Test cases for the vq-gan trainer."""

  def setUp(self):
    super().setUp()
    self.workdir = self.create_tempdir().full_path
    self.writer = metric_writers.create_default_writer(
        self.workdir, just_logging=True, asynchronous=True)

  def get_config(self, arch):
    if arch == '3dcnn':
      config = vqgan3d_ucf101_config.get_config('B-runlocal')
    else:
      config = vqgan2d_ucf101_config.get_config('B-runlocal')

    config.batch_size = 2
    config.eval_batch_size = config.get_ref('batch_size')
    config.debug_num_batches = 2
    config.num_training_epochs = 1

    config.pretrained_image_model = False
    config.perceptual_loss_weight = 0.0

    config.vqvae.filters = 32
    config.vqvae.num_enc_res_blocks = 1
    config.vqvae.num_dec_res_blocks = 1
    config.vqvae.channel_multipliers = (1,) * len(
        config.vqvae.channel_multipliers)
    config.discriminator.channel_multipliers = (1,) * len(
        config.discriminator.channel_multipliers)

    config.lr_configs.steps_per_cycle = config.get_ref(
        'num_training_epochs') * config.get_ref('debug_num_batches')
    config.lr_configs.warmup_steps = 0

    if 'init_from' in config:
      del config.init_from

    config.eval_from.step = 0
    config.eval.num_examples = 4
    config.eval.final_num_repeats = 2
    config.eval.enable_inception_score = False
    config.eval.enable_frechet_distance = False

    return config

  @parameterized.parameters('3dcnn', '2dcnn')
  def test_train(self, arch):
    config = self.get_config(arch)
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, data_rng = jax.random.split(rng)
    dataset = train_utils.get_pseudo_dataset(config, data_rng)
    train_state, _ = vqgan_trainer.train(
        rng=rng,
        config=config,
        dataset=dataset,
        workdir=self.workdir,
        writer=self.writer)
    del train_state
    logging.info('workdir content: %s', tf.io.gfile.listdir(self.workdir))

  def test_finetune(self):
    config = self.get_config('3dcnn')
    config.vqgan.finetune_decoder = True
    config.init_from = None    # disable the 3d inflation initialization
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, data_rng = jax.random.split(rng)
    dataset = train_utils.get_pseudo_dataset(config, data_rng)
    train_state, _ = vqgan_trainer.train(
        rng=rng,
        config=config,
        dataset=dataset,
        workdir=self.workdir,
        writer=self.writer)
    del train_state
    logging.info('workdir content: %s', tf.io.gfile.listdir(self.workdir))

  @parameterized.parameters('3dcnn', '2dcnn')
  def test_eval(self, arch):
    config = self.get_config(arch)
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, data_rng = jax.random.split(rng)
    dataset = train_utils.get_pseudo_dataset(config, data_rng)
    vqgan_trainer.evaluate(
        rng=rng,
        config=config,
        dataset=dataset,
        workdir=self.workdir,
        writer=self.writer)
    logging.info('workdir content: %s', tf.io.gfile.listdir(self.workdir))


if __name__ == '__main__':
  absltest.main()
