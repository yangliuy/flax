# Copyright 2020 The Flax Authors.
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
"""LSTM classifier model for SST-2."""

import functools
from typing import Any, Callable, Dict, Optional, Sequence, Union, Tuple

import flax
from flax import nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

# pylint: disable=arguments-differ,too-many-arguments

ShapeType = Tuple[int]
ShapeAndType = Tuple[ShapeType, np.dtype]


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def create_model(model_definition: Any,
                 shapes_and_types: Sequence[ShapeAndType],
                 model_kwargs: Dict[str, Any]) -> flax.nn.Model:
  """Helper function to instantiate a model.

  Args:
    model_definition: A Flax module to be instantiated.
    shapes_and_types: A sequence of (shape, dtype) tuples indicating the shapes
      and types of the inputs to the model.
    model_kwargs: A dictionary with keys/values to be passed to the model.

  Returns:
    The instantiated model with its parameters.
  """
  model_definition = model_definition.partial(**model_kwargs)
  rng = jax.random.PRNGKey(0)
  with flax.nn.stateful():
    _, params = model_definition.init_by_shape(rng, shapes_and_types)
  return flax.nn.Model(model_definition, params)


def sequence_mask(lengths: jnp.ndarray, max_length: int) -> jnp.ndarray:
  """Computes a boolean mask over sequence positions for each given length.

  Example:
  ```
  sequence_mask([1, 2], 3)
  [[True, False, False],
   [True, True, False]]
  ```

  Args:
    lengths: The length of each sequence. <int>[batch_size]
    max_length: The width of the boolean mask. Must be >= max(lengths).

  Returns:
    A mask with shape: <bool>[lengths.size, max_length] indicating which
    positions are valid for each sequence.
  """
  return jnp.arange(max_length) < jnp.expand_dims(lengths, 1)


def word_dropout(inputs: jnp.ndarray,
                 rate: float,
                 unk_idx: int,
                 deterministic: bool = False) -> jnp.ndarray:
  """Replaces a fraction (rate) of inputs with `unk_idx`.

  Args:
    inputs: A batch of token ID sequences <int>[batch_size, sequence_length].
    rate: The word dropout rate with which to replace token IDs with unk_idx.
    unk_idx: The ID for the unknown word (e.g., the ID for <unk>).
    deterministic: If True, do not apply word dropout and return the identity.

  Returns:
    The inputs with a fraction (`rate`) of token IDs replaced with `unk_idx`.
  """
  if deterministic or np.isclose(rate, 0.):
    return inputs
  mask = jax.random.bernoulli(nn.make_rng(), p=rate, shape=inputs.shape)
  return jnp.where(mask, jnp.array([unk_idx]), inputs)


class MLP(nn.Module):
  """A multi-layer MLP."""

  def apply(self,
            inputs: jnp.ndarray,
            hidden_sizes: Sequence[int],
            activation_fn: Callable[..., Any] = nn.tanh,
            dropout_rate: float = 0.,
            output_bias: bool = True,
            train: bool = False):
    """Applies a multi-layered perceptron (MLP) to the inputs.

    Applies `num_layers`-1 Dense layers with `activation_fn` and then a final
    Dense layer without activation for which the bias can be disabled using
    `output_bias`.

    Example:
    ```
    MLP(inputs, hidden_sizes=[10, 4], dropout_rate=0.5, output_bias=False)
    ```
    This applies a dense layer with 10 units, and then an output layer with 4
    units without a bias term. Since `activation_fn` was not specified, a
    tanh is applied to the intermediate layer.

    Args:
      inputs: The inputs to the MLP. <float32>[batch_size, ..., input_size].
      hidden_sizes: The hidden sizes of each layer, including the output layer.
      activation_fn: The activation function to apply to each layer except the
        final output layer.
      dropout_rate: The dropout rate applied in-between layers when training.
      output_bias: If False, do not use a bias term in the last layer.
      train: Enables dropout when True.

    Returns:
      The output of the MLP. <float32>[batch_size, ..., hidden_sizes[-1]].
    """
    assert np.all(np.array(hidden_sizes) > 0), \
        f'`hidden_sizes` was set to {hidden_sizes}, but each value must be >=1.'

    hidden = inputs
    for layer_idx, hidden_size in enumerate(hidden_sizes[:-1]):
      hidden = nn.Dense(hidden, hidden_size, name=f'layer_{layer_idx}')
      hidden = activation_fn(hidden)
      if train and dropout_rate > 0.:
        hidden = nn.dropout(hidden, rate=dropout_rate)

    output = nn.Dense(hidden, hidden_sizes[-1], bias=output_bias, name='output')
    return output


class Embedding(nn.Module):
  """Embeds batches of token IDs into feature space."""

  def apply(self,
            inputs: jnp.ndarray,
            num_embeddings: int,
            features: int,
            embedding_init: Callable[..., jnp.ndarray] = nn.initializers.normal(
                stddev=0.1),
            frozen: bool = False,
            dropout_rate: float = 0.,
            word_dropout_rate: float = 0.,
            unk_idx: Optional[int] = None,
            train: bool = False):
    """Embeds the input sequences.

    Args:
      inputs: Batch of input token ID sequences <int64>[batch_size, seq_length].
      num_embeddings: The number of embeddings (e.g., size of the vocabulary).
      features: The dimensionality of the embeddings.
      embedding_init: An initializer for the embeddings matrix.
      frozen: If True, block gradients to keep the embeddings unchanged.
      dropout_rate: Dropout probability to apply after embedding the inputs.
      word_dropout_rate: Probability with which to replace inputs with unk_idx.
      unk_idx: The unknown word ID. Must be provided when word_dropout > 0.
      train: Enables dropout and word dropout when True.

    Returns:
      The embedded inputs, shape: <float32>[batch_size, seq_length, features].
    """
    if train and word_dropout_rate > 0.:
      assert unk_idx is not None, 'Provide unk_idx when using word_dropout.'
      inputs = word_dropout(inputs, word_dropout_rate, unk_idx=unk_idx)

    embedding_matrix = self.param('embedding', (num_embeddings, features),
                                  embedding_init)
    embedded_inputs = jnp.take(embedding_matrix, inputs, axis=0)

    # Keep the embeddings fixed at initial (e.g. pretrained) values.
    if frozen:
      embedded_inputs = lax.stop_gradient(embedded_inputs)

    if train and dropout_rate > 0.:
      embedded_inputs = nn.dropout(embedded_inputs, rate=dropout_rate)

    return embedded_inputs


@jax.vmap
def flip_sequences(inputs: jnp.ndarray, lengths: jnp.ndarray) -> jnp.array:
  """Flips a sequence of inputs along the time dimension."""
  return jnp.flip(jnp.roll(inputs, inputs.shape[0] - lengths, axis=0), axis=0)


class LSTM(nn.Module):
  """LSTM encoder."""

  def apply(self, inputs: jnp.ndarray, lengths: jnp.ndarray, hidden_size: int,
            **lstm_cell_kwargs):
    # pylint: disable=unused-argument, arguments-differ
    # inputs.shape = <float32>[batch_size, seq_length, emb_size].
    # lengths.shape = <int64>[batch_size,]
    batch_size = inputs.shape[0]
    carry = nn.LSTMCell.initialize_carry(
        jax.random.PRNGKey(0), (batch_size,), hidden_size)
    _, outputs = flax.jax_utils.scan_in_dim(
        nn.LSTMCell.partial(name='lstm_cell', **lstm_cell_kwargs),
        carry,
        inputs,
        axis=1)
    final_states = outputs[jnp.arange(batch_size),
                           jnp.maximum(0, lengths - 1), :]
    return outputs, final_states


class BidirectionalLSTM(nn.Module):
  """Bidirectional LSTM encoder."""

  def apply(self, inputs: jnp.ndarray, lengths: jnp.ndarray, hidden_size: int,
            **lstm_cell_kwargs) -> jnp.ndarray:
    # pylint: disable=arguments-differ
    if hidden_size % 2 != 0:
      raise ValueError('Hidden size must be even.')
    forward, forward_final = LSTM(
        inputs,
        lengths,
        hidden_size // 2,
        name='forward_lstm',
        **lstm_cell_kwargs)
    flipped = flip_sequences(inputs, lengths)
    backward, backward_final = LSTM(
        flipped,
        lengths,
        hidden_size // 2,
        name='backward_lstm',
        **lstm_cell_kwargs)
    backward = flip_sequences(backward, lengths)
    outputs = jnp.concatenate((forward, backward), -1)
    final_states = [forward_final, backward_final]
    return outputs, final_states


class KeysOnlyMlpAttention(nn.Module):
  """MLP attention module that returns a scalar score for each key."""

  def apply(self,
            keys: jnp.ndarray,
            mask: jnp.ndarray,
            hidden_size: int = None,
            export_attention: bool = False) -> jnp.ndarray:
    """Computes MLP-based attention scores based on keys alone, without a query.

    Attention scores are computed by feeding the keys through an MLP. This
    results in a single scalar per key, and for each sequence the attention
    scores are normalized using a softmax so that they sum to 1. Invalid key
    positions are ignored as indicated by the mask. This is also called
    "Bahdanau attention" and was originally proposed in:
    ```
    Bahdanau et al., 2015. Neural Machine Translation by Jointly Learning to
    Align and Translate. ICLR. https://arxiv.org/abs/1409.0473
    ```
    This version that only uses keys was used in various papers, for example in
    ```
    Jain & Wallace, 2019. Attention is not Explanation. NAACL.
    https://www.aclweb.org/anthology/N19-1357/
    ```

    Args:
      keys: The inputs for which to compute an attention score. Shape:
        <float32>[batch_size, seq_length, keys_size].
      mask: A mask that determinines which values in `keys` are valid. Only
        values for which the mask is True will get non-zero attention scores.
        <bool>[batch_size, seq_length].
      hidden_size: The hidden size of the MLP that computes the attention score.
      export_attention: Set to True to save the computed attention to the state.

    Returns:
      The normalized attention scores. <float32>[batch_size, seq_length].
    """
    hidden = nn.Dense(keys, hidden_size, name='keys', bias=False)
    energy = nn.tanh(hidden)
    scores = nn.Dense(energy, 1, name='energy', bias=False)
    scores = scores.squeeze(-1)  # New shape: <float32>[batch_size, seq_len].
    scores = jnp.where(mask, scores, -jnp.inf)  # Using exp(-inf) = 0 below.
    scores = nn.softmax(scores, axis=-1)

    # Use the state to export the attention scores.
    attention_memory = self.state('attention')
    if not self.is_initializing() and export_attention:
      attention_memory.value = scores

    return scores  # Shape: <float32>[batch_size, seq_len]


class AttentionClassifier(nn.Module):
  """A classifier that uses attention to summarize the inputs."""

  def apply(self,
            encoded_inputs: jnp.ndarray,
            input_lengths: jnp.ndarray,
            hidden_size: int,
            output_size: int,
            dropout_rate: float = 0.,
            train: bool = False) -> jnp.ndarray:
    """Attends over the inputs, summarizes them, and makes a classification.

    Args:
      encoded_inputs: The inputs (e.g., sentences) that have already been
        encoded by some encoder, e.g., an LSTM. <float32>[batch_size,
        seq_length, encoded_inputs_size].
      input_lengths: The lengths of the inputs. <int64>[batch_size].
      hidden_size: The hidden size of the MLP classifier.
      output_size: The number of output classes for the classifier.
      dropout_rate: The dropout rate applied over the encoded_inputs, the
        summary of the inputs, and inside the MLP. Applied when `train` is True.
      train: Enables dropout when True.

    Returns:
      An array of logits <float32>[batch_size, output_size].
    """
    if train:
      encoded_inputs = nn.dropout(encoded_inputs, rate=dropout_rate)

    # Compute attention. attention.shape: <float32>[batch_size, seq_len].
    mask = sequence_mask(input_lengths, encoded_inputs.shape[1])
    attention = KeysOnlyMlpAttention(
        encoded_inputs,
        mask,
        hidden_size=hidden_size,
        export_attention=not train)

    # Summarize the inputs by taking their weighted sum using attention scores.
    context = jnp.expand_dims(attention, 1) @ encoded_inputs
    context = context.squeeze(1)  # <float32>[batch_size, encoded_inputs_size]
    if train:
      context = nn.dropout(context, rate=dropout_rate)

    # Make the final prediction from the context vector (the summarized inputs).
    logits = MLP(
        context,
        hidden_sizes=[hidden_size, output_size],
        output_bias=False,
        dropout_rate=dropout_rate,
        name='mlp',
        train=train)
    return logits


class TextClassifier(nn.Module):
  """A general-purpose text classification model.

  For each module_method `f`, there is a private method `_f`. This is because we
  cannot call module methods directly from apply(). To be able to use the same
  function in apply(), and also expose it as model.f(), we need to define a
  shared function `_f` called by both `apply(...)` and the module method `f`.
  """

  def _embed(self,
             embedder: Any,
             inputs: jnp.ndarray,
             train: bool = False) -> jnp.ndarray:
    return embedder(inputs, train=train, name='embedder')

  @nn.module_method
  def embed(self,
            inputs: jnp.ndarray,
            train: bool = False,
            embedder: Any = None,
            **unused_kwargs) -> jnp.ndarray:
    """Embeds the inputs in feature space using the provided embedder."""
    # Module methods are passed all apply(..) arguments, hence **unused_kwargs.
    assert embedder is not None, '`embedder` must be provided to `embed(...)`.'
    return self._embed(embedder, inputs, train=train)

  def _encode(self,
              encoder: Any,
              embedded_inputs: jnp.ndarray,
              input_lengths: jnp.ndarray,
              train: bool = False) -> jnp.ndarray:
    return encoder(embedded_inputs, input_lengths, train=train, name='encoder')

  @nn.module_method
  def encode(self,
             embedded_inputs: jnp.ndarray,
             input_lengths: jnp.ndarray,
             train: bool = False,
             encoder: bool = None,
             **unused_kwargs) -> Union[jnp.ndarray, Sequence[jnp.ndarray]]:
    """Encodes the embedded inputs using the provided encoder."""
    # Module methods are passed all apply(..) arguments, hence **unused_kwargs.
    assert encoder is not None, '`encoder` must be provided to `encode(...)``.'
    return self._encode(encoder, embedded_inputs, input_lengths, train=train)

  def _classify(self,
                classifier: Any,
                encoded_inputs: jnp.ndarray,
                input_lengths: jnp.ndarray,
                train: bool = False) -> jnp.ndarray:
    return classifier(
        encoded_inputs, input_lengths, train=train, name='classifier')

  @nn.module_method
  def classify(self,
               encoded_inputs: jnp.ndarray,
               input_lengths: jnp.ndarray,
               train: bool = False,
               classifier: Any = None,
               **unused_kwargs) -> jnp.ndarray:
    """Makes a classification from the provided encoded inputs."""
    # Module methods are passed all apply(..) arguments, hence **unused_kwargs.
    assert classifier is not None, \
        '`classifier` must be provided to `classify(...)`.'
    return self._classify(
        classifier, encoded_inputs, input_lengths, train=train)

  def apply(self,
            inputs: jnp.ndarray,
            input_lengths: jnp.ndarray,
            embedder: Any,
            encoder: Any,
            classifier: Any,
            train: bool = False) -> jnp.ndarray:
    """Embeds the inputs, encodes them, and then makes a classification.

    The role of the embedder is to take the input token IDs and embed them into
    feature space. The embedder may apply dropout on the resulting embeddings,
    and word dropout on the input token IDs.

    The role of the encoder is to take the embeddings and encode them in some
    way so as to e.g., capture more context. An example is a BiLSTM that
    conditions the representation of each token on its left and right context.
    The encoder can also simply be the identity or an average pooling operation.

    The role of the classifier is to take the encoded inputs and turn them into
    a classification output. Examples are an MLP or an AttentionClassifier.

    Args:
      inputs: The input token ID sequences. <int64>[batch_size, seq_length].
      input_lengths: The lengths of the input sequences. <int64>[batch_size].
      embedder: An embedder module that takes inputs and returns embeddings.
      encoder: An encoder module that takes embeddings and encodes them.
      classifier: A classifier module that makes a classification given encoded
        inputs.
      train: Set to True to enable dropout.

    Returns:
      The output logits <float32>[batch_size, num_classes].
    """
    embedded_inputs = self._embed(embedder, inputs, train=train)
    encoded_inputs, _ = self._encode(
        encoder, embedded_inputs, input_lengths, train=train)
    logits = self._classify(
        classifier, encoded_inputs, input_lengths, train=train)
    return logits
