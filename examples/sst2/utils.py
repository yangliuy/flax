# Lint as: python3
"""A collection of utility functions for text classification."""
import collections
from typing import Dict, Iterable, Sequence, Text
from absl import logging

import flax
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile

VocabType = Dict[bytes, int]


def cardinality(dataset: tf.data.Dataset) -> int:
  """Returns the number of examples in the dataset by iterating over it once."""
  return dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()


def build_tf_hashtable(vocabulary: VocabType,
                       unk_idx: int) -> tf.lookup.StaticHashTable:
  """Returns a TF lookup table from a vocabulary."""
  return tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          list(vocabulary.keys()), list(vocabulary.values())),
      default_value=unk_idx)


def save_embeddings(embeddings: np.ndarray, path: Text):
  """Saves Flax-serialized word embeddings to disk."""
  embedding_bytes = flax.serialization.to_bytes(embeddings)
  with gfile.GFile(path, 'wb') as f:
    f.write(embedding_bytes)
  logging.info('Saved embeddings to: %s', path)


def write_glove_embeddings(file_handler, glove_tokens, glove_vectors):
  """Saves GloVe tokens and embeddings in GloVe space-separated text format."""
  with file_handler as f:
    for token, vector in zip(glove_tokens, glove_vectors):
      vector_bytes = b' '.join([b'%.6f' % v for v in vector])
      f.write(token + b' ' + vector_bytes + b'\n')


def load_cached_embeddings(path: Text) -> np.ndarray:
  """Loads Flax-serialized word embeddings from disk."""
  with gfile.GFile(path, 'rb') as f:
    embeddings = flax.serialization.from_bytes(np.ndarray, f.read())
  logging.info('Loaded cached word embeddings, shape: %r', embeddings.shape)
  return embeddings


def load_glove_embeddings(vocab: VocabType, glove_path: Text,
                          embedding_size: int, seed: int) -> np.ndarray:
  """Initializes a word embedding matrix using GloVe pre-trained embeddings.

  The word embedding matrix is initialized randomly, and tokens that are found
  in the GloVe file are replaced with their GloVe pre-trained embeddings. Other
  embeddings remain random. The first embedding is assumed to be a padding token
  and is initialized with zeros.

  The GloVe file contains lines with the following format:

  ```
    cat 0.01 0.02 ... -0.02
    dog 0.02 0.03 ... -0.04
  ```
  That is, first the token, and then space-separated the value of each
  dimension. The number of characters each numeric value takes may differ.

  Args:
    vocab: The vocabulary to build embeddings for.
    glove_path: Path to GloVe word vectors.
    embedding_size: The size of the GloVe word embeddings. For GloVe 840B the
      embedding size is 300.
    seed: A seed to use to initialize the special token vectors.

  Returns:
    A tuple with the vocabulary (token -> ID) and the word embedding matrix.
  """
  num_found = 0

  # Initialize word embeddings randomly.
  embeddings = np.random.RandomState(seed).uniform(
      -0.1, 0.1, size=[len(vocab), embedding_size])
  embeddings[0] = np.zeros(embedding_size)  # Padding hardcoded at position 0.

  # Extract the embeddings for our set of tokens (discard the rest).
  with gfile.GFile(glove_path, mode='rb') as f:
    for line in f:
      token, vector_str = line.strip().split(maxsplit=1)
      if token in vocab:
        vector = np.fromstring(vector_str, sep=' ')
        assert vector.size == embedding_size, (f'Unexpected GloVe vector size '
                                               f'(found {vector.size}, expected'
                                               f' {embedding_size})')
        embeddings[vocab[token]] = vector
        num_found += 1

  logging.info('Found %d tokens in GloVe file.', num_found)
  return embeddings

