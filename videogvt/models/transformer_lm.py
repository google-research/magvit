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

"""Auto-regressive Transformers."""
from typing import Any, Callable, Sequence

from flax import linen as nn
from flax import struct
from jax import lax
import jax.numpy as jnp
import numpy as np

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  output_vocab_size: int
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 64
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 64
  mlp_dim: int = 128
  max_len: int = 64
  max_predict_length: int = 64
  input_dropout_rate: float = 0.1
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  # swish, swigleu would be more advanced.
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu  # relu
  decode: bool = False
  # it will increase the stablity set kernel_init to xavier((2**-0.5))
  # for attention layers
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.zeros
  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)


class MlpBlock(nn.Module):
  """MLP / feed-forward block."""
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs, deterministic):
    """Applies Transformer MlpBlock module."""
    config = self.config
    out_dim = inputs.shape[-1]
    x = nn.Dense(config.mlp_dim,
                 dtype=config.dtype,
                 kernel_init=config.kernel_init,
                 bias_init=config.bias_init)(inputs)
    x = config.activation_fn(x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=deterministic)
    output = nn.Dense(out_dim,
                      dtype=config.dtype,
                      kernel_init=config.kernel_init,
                      bias_init=config.bias_init)(x)
    return output


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class PositionEmbeddings(nn.Module):
  """Learnable positional embeddings to the inputs."""
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs):
    config = self.config
    length = inputs.shape[1]
    posemb_shape = (1, config.max_len, config.emb_dim)

    if config.posemb_init is None:
      pos_embedding = sinusoidal_init(max_len=config.max_len)(
          None, posemb_shape)
    else:
      pos_embedding = self.param('embed_pos', config.posemb_init, posemb_shape)

    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    # did some preparation work for autogressive decoding.
    if config.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        cur_index = cache_index.value
        cache_index.value = cur_index + length
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, cur_index, 0)),
                               (1, length, config.emb_dim))
    return inputs + pe


class TransformerBlock(nn.Module):
  """Transformer block.

  Attributes:
    config: ml_collections.ConfigDict dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs, deterministic, attn_mask=None):
    """Applies Transformer module.

    Args:
      inputs: input data.
      deterministic: deterministic
      attn_mask: self-attention mask.

    Returns:
      output from VTNLayer: [batch_size, seq_length, model_dim].
    """
    config = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=config.dtype)(inputs)
    x = nn.SelfAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=deterministic,
        decode=config.decode)(x, attn_mask)

    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=deterministic)
    x = x + inputs
    # MLP block.
    y = nn.LayerNorm(dtype=config.dtype)(x)
    y = MlpBlock(config)(y, deterministic=deterministic)
    y = nn.Dropout(rate=config.dropout_rate)(
        y, deterministic=deterministic)

    return x + y


class TransformerLM(nn.Module):
  """Transformer LM."""
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs, deterministic=True):
    """Applies Transformer decoder on the inputs.

    Args:
      inputs: input data
      deterministic: deterministic

    Returns:
      output: [batch_size, seq_length, model_dim]
    """
    config = self.config
    x = inputs.astype('int32')

    if config.decode:
      # decoder_mask will be defined in multiheadattention
      decoder_mask = None
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(inputs > 0, inputs > 0, dtype=config.dtype),
          nn.make_causal_mask(inputs, dtype=config.dtype))

    # Input Embedding
    input_embed = nn.Embed(
        num_embeddings=config.vocab_size,
        features=config.emb_dim,
        embedding_init=nn.initializers.normal(stddev=config.emb_dim**-0.5))

    x = input_embed(x)
    x = nn.Dropout(rate=config.input_dropout_rate)(
        x, deterministic=deterministic)
    x = PositionEmbeddings(config)(x)

    # Decoder
    for layer in range(config.num_layers):
      x = TransformerBlock(config, name=f'VTNLayer_{layer}')(
          x, deterministic=deterministic, attn_mask=decoder_mask)

    decoder_out = nn.LayerNorm(dtype=config.dtype, name='decoder_out_ln')(x)

    if config.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      assert config.vocab_size == config.output_vocab_size
      logits = input_embed.attend(decoder_out.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(decoder_out.shape[-1])
    else:
      logits = nn.Dense(
          config.output_vocab_size,
          dtype=config.dtype,
          kernel_init=nn.initializers.normal(
              stddev=decoder_out.shape[-1]**-0.5),
          bias_init=config.bias_init,
          name='logits_dense')(decoder_out)
    return logits

