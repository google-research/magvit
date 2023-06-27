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

"""Flaxformer implementation for conditional and unconditional BERT."""
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp

from flaxformer.architectures.t5 import t5_architecture
from flaxformer.components import dense
from flaxformer.components import layer_norm
from flaxformer.components.attention import dense_attention

InitializerType = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]

# which activation fn to use
ACTIVATION = jax.nn.gelu


def truncated_normal(stddev: Union[float, jnp.ndarray], dtype=jnp.float32):

  def init(key: jnp.ndarray, shape: Sequence[int], dtype: jnp.dtype = dtype):
    return jax.random.truncated_normal(
        key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev

  return init


class Bias(nn.Module):
  """Adds a (learned) bias to the input.


  Attributes:
    dtype: the dtype of the computation (default: float32).
    bias_init: initializer function for the bias.
  """
  dtype: Any = jnp.float32
  bias_init: Callable[[Any, Tuple[int], Any], Any] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)

    bias_shape = inputs.shape[-1]
    bias = self.param('bias', self.bias_init, bias_shape)
    bias = jnp.asarray(bias, self.dtype)
    bias = jnp.broadcast_to(bias, inputs.shape)

    return inputs + bias


class BertEmbed(nn.Module):
  """Embeds Bert-style."""
  embedding_size: int
  hidden_dropout_prob: float
  vocab_size: int
  max_position_embeddings: int
  num_segments: int
  initializer_fn: InitializerType
  hidden_size: Optional[int] = None

  @nn.compact
  def __call__(self, input_ids: jnp.ndarray, segment_ids: Optional[jnp.ndarray],
               *, deterministic: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    seq_length = input_ids.shape[-1]
    position_ids = jnp.arange(seq_length)[None, :]

    word_embedder = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='word_embeddings')
    word_embeddings = word_embedder(input_ids)
    position_embeddings = nn.Embed(
        num_embeddings=self.max_position_embeddings,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='position_embeddings')(
            position_ids)
    if self.num_segments > 0:
      segment_embeddings = nn.Embed(
          num_embeddings=self.num_segments,
          features=self.embedding_size,
          embedding_init=self.initializer_fn,
          name='segment_embeddings')(
              segment_ids)
    else:
      segment_embeddings = 0

    input_embeddings = nn.LayerNorm(
        name='embeddings_ln', use_bias=False)(
            word_embeddings + position_embeddings + segment_embeddings)
    if self.hidden_size:
      input_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=self.initializer_fn,
          name='embedding_hidden_mapping',
          use_bias=False)(
              input_embeddings)
    input_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        input_embeddings, deterministic=deterministic)

    return input_embeddings


class BertEmbProject(nn.Module):
  """Embeds Bert-style."""
  embedding_size: int
  hidden_dropout_prob: float
  max_position_embeddings: int
  initializer_fn: InitializerType
  hidden_size: Optional[int] = None

  @nn.compact
  def __call__(self, input_emb: jnp.ndarray,
               deterministic: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    seq_length = input_emb.shape[1]
    position_ids = jnp.arange(seq_length)[None, :]
    proj_embedder = nn.Dense(
        features=self.embedding_size,
        kernel_init=self.initializer_fn,
        name='cond_embeddings')
    proj_embeddings = proj_embedder(input_emb)
    position_embeddings = nn.Embed(
        num_embeddings=self.max_position_embeddings,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='position_embeddings')(
            position_ids)
    proj_embeddings = nn.LayerNorm(
        name='embeddings_ln', use_bias=False)(
            proj_embeddings + position_embeddings)
    if self.hidden_size:
      proj_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=self.initializer_fn,
          name='embedding_hidden_mapping',
          use_bias=False)(
              proj_embeddings)
    proj_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        proj_embeddings, deterministic=deterministic)
    return proj_embeddings


class BertMlmLayer(nn.Module):
  """Bert layer for masked token prediction."""
  hidden_size: int
  initializer_fn: Optional[InitializerType]

  @nn.compact
  def __call__(self, last_layer: jnp.ndarray,
               embeddings: jnp.ndarray) -> jnp.ndarray:
    mlm_hidden = nn.Dense(
        features=self.hidden_size, name='mlm_dense', use_bias=False)(
            last_layer)
    mlm_hidden = ACTIVATION(mlm_hidden)
    mlm_hidden = nn.LayerNorm(name='mlm_ln', use_bias=False)(mlm_hidden)
    output_weights = jnp.transpose(embeddings)
    logits = jnp.matmul(mlm_hidden, output_weights)
    logits = Bias(name='mlm_bias')(logits)
    return logits


class Bert(nn.Module):
  """BERT as a Flax module."""
  vocab_size: int
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 512
  num_segments: int = 0
  initializer_range: float = 0.02
  pad_token_id: int = -1

  @nn.compact
  def __call__(self,
               input_ids: jnp.ndarray,
               segment_ids: Optional[jnp.ndarray] = None,
               *,
               deterministic: bool = True) -> jnp.ndarray:
    # We assume that all pad tokens should be masked out.
    input_ids = input_ids.astype('int32')

    input_embeddings = BertEmbed(
        embedding_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        vocab_size=self.vocab_size,
        max_position_embeddings=self.max_position_embeddings,
        num_segments=self.num_segments,
        initializer_fn=truncated_normal(self.initializer_range))(
            input_ids=input_ids,
            segment_ids=segment_ids,
            deterministic=deterministic)

    # Stack BERT layers.
    layer_input = input_embeddings
    use_bias = False

    def dropout_factory():
      return nn.Dropout(rate=self.hidden_dropout_prob)

    def self_attention_factory():
      init_kwargs = dict(
          broadcast_dropout=False,
          dropout_rate=self.attention_probs_dropout_prob,
          qkv_features=self.hidden_size,
          num_heads=self.num_attention_heads,
          rescale_logits=True,
          use_bias=use_bias)
      return dense_attention.MultiHeadDotProductAttention(**init_kwargs)

    def mlp_factory():
      init_kwargs = dict(
          final_dropout_rate=0,
          intermediate_dim=self.intermediate_size,
          intermediate_dropout_rate=self.hidden_dropout_prob,
          activations=(ACTIVATION,),
          use_bias=use_bias)
      return dense.MlpBlock(**init_kwargs)

    input_mask = (input_ids != self.pad_token_id).astype(jnp.int32)
    decoder_mask = nn.make_attention_mask(input_mask, input_mask)

    for lyr in range(self.num_hidden_layers):
      layer_output = t5_architecture.DecoderLayer(
          self_attention=self_attention_factory(),
          mlp=mlp_factory(),
          encoder_decoder_attention=self_attention_factory(),
          dropout_factory=dropout_factory,
          layer_norm_factory=layer_norm.T5LayerNorm,
          name=f'flaxformer_decoder_layer_{lyr}')(
              targets=layer_input,
              encoded=None,
              enable_dropout=self.attention_probs_dropout_prob and
              not deterministic,
              decoder_mask=decoder_mask,
          )
      layer_input = layer_output
    word_embedding_matrix = self.variables['params']['BertEmbed_0'][
        'word_embeddings']['embedding']
    logits = BertMlmLayer(
        hidden_size=self.hidden_size, initializer_fn=None)(
            last_layer=layer_output, embeddings=word_embedding_matrix)
    return logits
