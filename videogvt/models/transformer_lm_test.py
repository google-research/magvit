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

"""Tests for transformer."""

from absl.testing import absltest
import jax
import numpy as np
from videogvt.models import transformer_lm


class TransformerLMTest(absltest.TestCase):

  def get_config(self):
    config = transformer_lm.TransformerConfig(
        vocab_size=12,
        output_vocab_size=12)
    return config

  def get_one_batch(self):
    config = self.get_config()
    batch = np.random.randint(
        0, high=config.vocab_size, size=(2, config.max_len))
    return batch

  def test_train_one_step(self):
    config = self.get_config()
    model = transformer_lm.TransformerLM(config)
    rng = jax.random.PRNGKey(0)
    batch = self.get_one_batch()
    params = model.init({'params': rng, 'dropout': rng}, batch)
    outputs = model.apply(params, batch, rngs={'dropout': rng})
    outputs = jax.device_get(outputs)
    self.assertEqual(outputs.shape,
                     (2, config.max_len, config.output_vocab_size))

  def test_decode_one_step(self):
    config = self.get_config()
    predict_config = config.replace(decode=True)
    model = transformer_lm.TransformerLM(predict_config)
    rng = jax.random.PRNGKey(0)
    batch = self.get_one_batch()
    params = model.init(rng, batch)
    test_input = batch[:, :1]
    outputs = model.apply(
        params, test_input, mutable=['cache'], deterministic=True)
    outputs = jax.device_get(outputs)
    self.assertEqual(outputs[0].shape,
                     (2, 1, config.output_vocab_size))
    self.assertEqual(
        outputs[1]['cache']['VTNLayer_0']['SelfAttention_0']['cache_index'],
        [1])

if __name__ == '__main__':
  absltest.main()
