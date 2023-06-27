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

"""Simplified BERT."""

import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Text, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

# BERT layer norm
TF_LAYERNORM_EPSILON = 1e-12

InitializerType = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def truncated_normal(stddev: Union[float, jnp.ndarray], dtype=jnp.float32):
  def init(key: jnp.ndarray, shape: Sequence[int], dtype: jnp.dtype = dtype):
    return (
        jax.random.truncated_normal(
            key=key, lower=-2, upper=2, shape=shape, dtype=dtype
        )
        * stddev
    )

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


class BertAttention(nn.Module):
  """BERT attention layer that is part of each BERT layer."""

  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  hidden_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(
      self,
      layer_input: jnp.ndarray,
      input_mask: jnp.ndarray,
      deterministic: bool,
  ) -> jnp.ndarray:
    attention_mask = nn.make_attention_mask(input_mask, input_mask)
    attention_output = nn.attention.SelfAttention(
        num_heads=self.num_attention_heads,
        qkv_features=self.hidden_size,
        dropout_rate=self.attention_probs_dropout_prob,
        deterministic=deterministic,
        kernel_init=self.initializer_fn,
        bias_init=jax.nn.initializers.zeros,
        name='self_attention',
    )(layer_input, attention_mask)

    attention_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        attention_output, deterministic=deterministic
    )
    attention_output = nn.LayerNorm(
        epsilon=TF_LAYERNORM_EPSILON, name='attention_output_ln'
    )(attention_output + layer_input)

    return attention_output


class BertMlp(nn.Module):
  """BERT MLP layer that is part of each BERT layer."""

  hidden_size: int
  hidden_dropout_prob: float
  intermediate_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(
      self, attention_output: jnp.ndarray, deterministic: bool
  ) -> jnp.ndarray:
    # BERT intermediate layer.
    intermediate_output = nn.Dense(
        features=self.intermediate_size,
        kernel_init=self.initializer_fn,
        name='intermediate_output',
    )(attention_output)
    intermediate_output = jax.nn.gelu(intermediate_output)

    # BERT output layer.
    layer_output = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='layer_output',
    )(intermediate_output)
    layer_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        layer_output, deterministic=deterministic
    )
    layer_output = nn.LayerNorm(
        epsilon=TF_LAYERNORM_EPSILON, name='layer_output_ln'
    )(layer_output + attention_output)

    return layer_output


class BertLayer(nn.Module):
  """A single BERT layer."""

  intermediate_size: int
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(
      self,
      layer_input: jnp.ndarray,
      input_mask: jnp.ndarray,
      deterministic: bool,
  ) -> jnp.ndarray:
    attention_output = BertAttention(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        num_attention_heads=self.num_attention_heads,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_fn=self.initializer_fn,
    )(
        layer_input=layer_input,
        input_mask=input_mask,
        deterministic=deterministic,
    )

    layer_output = BertMlp(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        intermediate_size=self.intermediate_size,
        initializer_fn=self.initializer_fn,
    )(attention_output=attention_output, deterministic=deterministic)

    return layer_output


class BertEmbed(nn.Module):
  """Embeds BERT-style."""
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
        epsilon=TF_LAYERNORM_EPSILON, name='embeddings_ln')(
            word_embeddings + position_embeddings + segment_embeddings)
    if self.hidden_size:
      input_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=self.initializer_fn,
          name='emb_hidden_mapping')(
              input_embeddings)
    input_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        input_embeddings, deterministic=deterministic)

    return input_embeddings


class BertMlmLayer(nn.Module):
  """BERT layer for masked token prediction."""

  hidden_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(
      self, last_layer: jnp.ndarray, embeddings: jnp.ndarray
  ) -> jnp.ndarray:
    mlm_hidden = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='mlm_dense',
    )(last_layer)
    mlm_hidden = jax.nn.gelu(mlm_hidden)
    mlm_hidden = nn.LayerNorm(epsilon=TF_LAYERNORM_EPSILON, name='mlm_ln')(
        mlm_hidden
    )
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
               deterministic: bool = True) -> Dict[Text, jnp.ndarray]:
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
    for _ in range(self.num_hidden_layers):
      layer_output = BertLayer(  # pytype: disable=wrong-arg-types  # jax-ndarray
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=truncated_normal(self.initializer_range))(
              layer_input=layer_input,
              input_mask=jnp.ones_like(input_ids, dtype=jnp.int32),
              deterministic=deterministic)
      layer_input = layer_output

    word_embedding_matrix = self.variables['params']['BertEmbed_0'][
        'word_embeddings']['embedding']
    logits = BertMlmLayer(
        hidden_size=self.hidden_size,
        initializer_fn=truncated_normal(self.initializer_range))(
            last_layer=layer_output, embeddings=word_embedding_matrix)

    return logits  # pytype: disable=bad-return-type  # jax-ndarray


class BertCrossAttention(nn.Module):
  """BERT cross attention layer that is part of each BERT layer."""

  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  hidden_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(
      self,
      layer_input: jnp.ndarray,
      cond_input: jnp.ndarray,
      input_mask: jnp.ndarray,
      deterministic: bool,
  ) -> jnp.ndarray:
    inputs_q = layer_input
    inputs_kv = jnp.concatenate((cond_input, layer_input), axis=-2)
    attention_mask = nn.make_attention_mask(
        input_mask, jnp.ones_like(inputs_kv[..., 0])
    )
    attention_output = nn.attention.MultiHeadDotProductAttention(
        num_heads=self.num_attention_heads,
        qkv_features=self.hidden_size,
        out_features=self.hidden_size,
        dropout_rate=self.attention_probs_dropout_prob,
        deterministic=deterministic,
        kernel_init=self.initializer_fn,
        bias_init=jax.nn.initializers.zeros,
        name='self_attention',
    )(inputs_q, inputs_kv, attention_mask)
    attention_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        attention_output, deterministic=deterministic
    )
    attention_output = nn.LayerNorm(
        epsilon=TF_LAYERNORM_EPSILON, name='attention_output_ln'
    )(attention_output + layer_input)
    return attention_output


class CondBertLayer(nn.Module):
  """A single BERT layer."""

  intermediate_size: int
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(
      self,
      layer_input: jnp.ndarray,
      cond_input: jnp.ndarray,
      input_mask: jnp.ndarray,
      deterministic: bool,
  ) -> jnp.ndarray:
    attention_output = BertCrossAttention(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        num_attention_heads=self.num_attention_heads,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_fn=self.initializer_fn,
    )(
        layer_input=layer_input,
        cond_input=cond_input,
        input_mask=input_mask,
        deterministic=deterministic,
    )

    layer_output = BertMlp(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        intermediate_size=self.intermediate_size,
        initializer_fn=self.initializer_fn,
    )(attention_output=attention_output, deterministic=deterministic)

    return layer_output


