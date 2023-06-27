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

"""Tests for vq_models."""

from typing import Union

from absl.testing import parameterized
import jax
import ml_collections
import numpy as np
import tensorflow as tf
from videogvt.models import stylegan_discriminator_2d
from videogvt.models import stylegan_discriminator_2plus1d
from videogvt.models import stylegan_discriminator_3d
from videogvt.models import vqvae


class VqvaeTest(tf.test.TestCase, parameterized.TestCase):

  DISCRIMINATORS = {
      '2dcnn': stylegan_discriminator_2d,
      '3dcnn': stylegan_discriminator_3d,
      '2plus1dcnn': stylegan_discriminator_2plus1d
  }

  def get_config(
      self, arch: str, quantizer_cls: str = 'VectorQuantizer'
  ):
    config = ml_collections.ConfigDict()
    config.image_size = 32
    config.vqvae = ml_collections.ConfigDict()
    config.vqvae.architecture = arch
    config.vqvae.vector_quantizer_class = quantizer_cls
    config.vqvae.codebook_size = 128
    config.vqvae.entropy_loss_ratio = 0.1
    config.vqvae.entropy_temperature = 0.01
    config.vqvae.entropy_loss_type = 'softmax'
    config.vqvae.commitment_cost = 0.25
    config.vqvae.filters = 32
    config.vqvae.num_enc_res_blocks = 1
    config.vqvae.num_dec_res_blocks = 1
    config.vqvae.channel_multipliers = (1, 1)
    if arch != '2dcnn':
      config.vqvae.temporal_downsample = (True,)
    config.vqvae.embedding_dim = 32
    config.vqvae.conv_downsample = False
    config.vqvae.deconv_upsample = False
    config.vqvae.activation_fn = 'swish'
    config.vqvae.norm_type = 'GN'
    if arch == '2plus1dcnn':
      config.vqvae.expand_mid_features = False
    config.discriminator = ml_collections.ConfigDict()
    config.discriminator.filters = config.vqvae.get_oneway_ref('filters')
    config.discriminator.channel_multipliers = (1, 1, 1)
    if arch == '2plus1dcnn':
      config.discriminator.expand_mid_features = False
    return config

  def get_batch_iter(self):
    while True:
      # (batch, T, H, W, C)
      batch = np.random.uniform(size=(1, 2, 32, 32, 3)).astype(np.float32)
      yield batch

  @parameterized.parameters('2dcnn', '3dcnn', '2plus1dcnn')
  def test_vqvae(self, arch):
    rng = jax.random.PRNGKey(0)
    batch_iter = self.get_batch_iter()
    batch = next(batch_iter)
    config = self.get_config(arch)
    model = vqvae.VQVAE(config)
    params = model.init(rng, batch)
    outputs, result_dict = model.apply(params, batch)
    outputs = jax.device_get(outputs)
    self.assertShapeEqual(outputs, batch)
    self.assertEqual(result_dict['encodings'].shape[0], batch.shape[0])
    self.assertEqual(result_dict['encodings'].shape[-1],
                     config.vqvae.codebook_size)


  @parameterized.parameters('2dcnn', '3dcnn')
  def test_discriminator(self, arch):
    rng = jax.random.PRNGKey(0)
    batch_iter = self.get_batch_iter()
    batch = next(batch_iter)
    config = self.get_config(arch)
    model = self.DISCRIMINATORS[arch].StyleGANDiscriminator(config)
    params = model.init(rng, batch)
    outputs = model.apply(params, batch)
    if arch == '2dcnn':
      self.assertEqual(outputs.shape, (*batch.shape[:2], 1))
    else:
      self.assertEqual(outputs.shape, (batch.shape[0], 1))





if __name__ == '__main__':
  tf.test.main()
