# Copyright 2021 The BigBird Authors.
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

"""The main BigBird model and related functions."""

import copy

from absl import logging
from bigbird.core import decoder
from bigbird.core import encoder
from bigbird.core import utils
import tensorflow.compat.v2 as tf


class BertModel(tf.keras.layers.Layer):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into SentencePiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  params = utils.BigBirdConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(params, train=True)

  _, pooled_output = model(input_ids=input_ids, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self, params):
    """Constructor for BertModel.

    Args:
      params: `BigBirdConfig` dictionary.
    """
    self.params = copy.deepcopy(params)
    self.scope = params["scope"]
    super(BertModel, self).__init__(name=self.scope)

    # validate params
    self.pad = lambda x: x
    if params["max_encoder_length"] <= 512:
      logging.info("Switching to full attention for short sequences")
      self.params["attention_type"] = "original_full"
    if self.params["attention_type"] == "simulated_sparse" or self.params[
        "attention_type"] == "block_sparse":
      if params["max_encoder_length"] % params["block_size"]:
        logging.info("Expand max_encoder_length to next multiple of block_size")
        self.params["max_encoder_length"] = (
            params["max_encoder_length"] // params["block_size"] +
            1) * params["block_size"]
        pad_size = self.params["max_encoder_length"] - params[
            "max_encoder_length"]
        paddings = [[0, 0], [0, pad_size]]
        self.pad = lambda x: tf.pad(x, paddings)

    with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
      self.embeder = utils.EmbeddingLayer(
          vocab_size=self.params["vocab_size"],
          emb_dim=self.params["hidden_size"],
          initializer=utils.create_initializer(
              self.params["initializer_range"]),
          scale_emb=self.params["rescale_embedding"],
          use_token_type=True,
          num_token_types=self.params["type_vocab_size"],
          use_position_embeddings=True,
          max_position_embeddings=self.params["max_position_embeddings"],
          dropout_prob=self.params["hidden_dropout_prob"])
      self.encoder = encoder.EncoderStack(self.params)
      self.pooler = utils.SimpleDenseLayer(
          input_size=self.params["hidden_size"],
          output_size=self.params["hidden_size"],
          initializer=utils.create_initializer(
              self.params["initializer_range"]),
          activation=tf.tanh,
          name="pooler/dense")

  def call(self,
           input_ids,
           token_type_ids=None,
           training=None):
    """Constructor for BertModel.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      training: Boolean indicating whether the call is training or inference.

    Returns:
      sequence_output: Tensor of shape [batch_size, seq_length, hidden_size]
      pooled_output: Tensor of shape [batch_size, hidden_size]

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    # pad if needed
    input_ids = self.pad(input_ids)

    if token_type_ids is None:
      token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)
    else:
      token_type_ids = self.pad(token_type_ids)

    # Perform embedding lookup on the word ids.
    embedding_output = self.embeder(input_ids,
                                    self.params["max_encoder_length"],
                                    token_type_ids=token_type_ids,
                                    training=training)

    # Generate mask.
    input_mask = tf.where(input_ids > 0,
                          tf.ones_like(input_ids), tf.zeros_like(input_ids))

    # Run the stacked transformer.
    sequence_output = self.encoder(embedding_output, input_mask, training)

    # The "pooler" converts the encoded sequence tensor of shape
    # [batch_size, seq_length, hidden_size] to a tensor of shape
    # [batch_size, hidden_size]. This is necessary for segment-level
    # (or segment-pair-level) classification tasks where we need a fixed
    # dimensional representation of the segment.
    first_token_tensor = sequence_output[:, 0, :]
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. We assume that this has been pre-trained
    pooled_output = self.pooler(first_token_tensor)

    return sequence_output, pooled_output


class TransformerModel(tf.keras.layers.Layer):
  """Encoder-Decoder transformer model.

  Example usage:

  ```python
  # Already been converted into SentencePiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  target_ids = tf.constant([[43, 76, 38], [56, 8, 0]])

  params = utils.BigBirdConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.TransformerModel(params, train=True)

  predictions, _ = model(input_ids=input_ids, target_ids=target_ids)

  log_probs, logits, pred_ids = predictions
  ...
  ```
  """

  def __init__(self, params):
    """Constructor for TransformerModel.

    Args:
      params: `BigBirdConfig` dictionary.
    """
    self.params = copy.deepcopy(params)
    self.scope = params["scope"]
    super(TransformerModel, self).__init__(name=self.scope)

    # validate params
    self.pad = lambda x: x
    if params["max_encoder_length"] <= 512:
      logging.info("Switching to full attention for short sequences")
      self.params["attention_type"] = "original_full"
    if self.params["attention_type"] == "simulated_sparse" or self.params[
        "attention_type"] == "block_sparse":
      if params["max_encoder_length"] % params["block_size"]:
        logging.info("Expand max_encoder_length to next multiple of block_size")
        self.params["max_encoder_length"] = (
            params["max_encoder_length"] // params["block_size"] +
            1) * params["block_size"]
        pad_size = self.params["max_encoder_length"] - params[
            "max_encoder_length"]
        paddings = [[0, 0], [0, pad_size]]
        self.pad = lambda x: tf.pad(x, paddings)

    with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
      self.embeder = utils.EmbeddingLayer(
          vocab_size=self.params["vocab_size"],
          emb_dim=self.params["hidden_size"],
          initializer=utils.create_initializer(
              self.params["initializer_range"]),
          scale_emb=self.params["rescale_embedding"],
          use_token_type=False,
          num_token_types=None,
          use_position_embeddings=True,
          max_position_embeddings=self.params["max_position_embeddings"],
          dropout_prob=self.params["hidden_dropout_prob"])
      self.encoder = encoder.EncoderStack(self.params)
      self.decoder = decoder.DecoderStack(self.params)

  def _encode(self, input_ids, training=None):
    """Generate continuous representation for ids.

    Args:
      input_ids: Int tensor with shape [batch_size, input_length].
      training: Boolean indicating whether the call is training or inference.

    Returns:
      A float tensors of shape
          [batch_size, input_length, hidden_size].
    """
    # pad if needed
    input_ids = self.pad(input_ids)

    # Perform embedding lookup on the word ids.
    input_embs = self.embeder(
        input_ids, self.params["max_encoder_length"], training=training)

    # Generate mask.
    input_mask = tf.where(input_ids > 0,
                          tf.ones_like(input_ids), tf.zeros_like(input_ids))

    # Run the stacked transformer.
    encoder_output = self.encoder(input_embs, input_mask, training=training)

    return encoder_output, input_mask

  def _get_start_token_ids(self, tensor_for_shape):
    start_token_id = 2
    batch_size = utils.get_shape_list(tensor_for_shape)[0]
    return tf.ones([batch_size], dtype=tf.int32) * start_token_id

  def get_inputs_from_targets(self, targets, start_token_ids):
    """Converts target ids to input ids, i.e. adds <s> and removes last."""
    length = tf.math.count_nonzero(targets, axis=1, dtype=tf.int32)
    # Add start token ids.
    inputs = tf.concat([tf.expand_dims(start_token_ids, axis=1), targets], 1)
    # Remove </s> from the input.
    mask = tf.sequence_mask(length, self.params["max_decoder_length"]+1,
                            dtype=tf.int32)
    inputs = (mask * inputs)[:, :-1]
    return inputs

  def _decode(self, target_ids, target_mask, start_token_ids,
              encoder_output, encoder_mask, training=None):
    """Compute likelihood of target tokens under the model.

    Args:
      target_ids: tensor with shape [batch_size, target_length, hidden_size]
      target_mask: self-attention bias for decoder attention layer. [batch_size,
        input_length]
      start_token_ids: int32 tensor of shape [batch_size] for first decoder
        input.
      encoder_output: Continuous representation of input sequence. Float tensor
        with shape [batch_size, input_length, hidden_size].
      encoder_mask: Float tensor with shape [batch_size, input_length].
      training: Boolean indicating whether the call is training or inference.

    Returns:
      A dict containing the output ids, the output log-probs, the output logits.
    """

    # Prepare inputs to decoder layers by shifting targets, embedding ids,
    # adding positional encoding and applying dropout.
    input_ids = self.get_inputs_from_targets(target_ids, start_token_ids)

    input_embs = self.embeder(input_ids, self.params["max_decoder_length"],
                              training=training)

    outputs = self.decoder(input_embs, target_mask,
                           encoder_output, encoder_mask, training=training)

    logits = self.embeder.linear(outputs)
    output_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

    log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_ids, logits=logits)
    log_probs = tf.where(target_ids > 0, log_probs,
                         tf.zeros_like(log_probs, tf.float32))

    return (tf.identity(log_probs, name="log_probs"),
            tf.identity(logits, name="logits"),
            tf.cast(output_ids, tf.int32, name="pred_ids"),)

  def _init_cache(self, batch_size):
    """Initialize cache for decoding."""

    max_decode_len = self.params["max_decoder_length"]
    num_heads = self.params["num_attention_heads"]
    head_size = int(self.params["hidden_size"] / num_heads)

    cache = {}
    for layer in range(self.params["num_hidden_layers"]):
      cache["layer_%d" % layer] = {
          "k": tf.zeros([batch_size, num_heads, max_decode_len, head_size]),
          "v": tf.zeros([batch_size, num_heads, max_decode_len, head_size]),
      }
    return cache

  def _get_symbols_to_logits_fn(self, decoder_self_attention_mask):
    """Returns a decoding function that calculates logits of the next tokens."""

    max_decode_len = self.params["max_decoder_length"]

    def _symbols_to_logits_fn(target_ids, cache, i):
      """Generate logits for next candidate IDs.

      Args:
        target_ids: Current decoded sequences. int tensor with shape
          [batch_size, i + 1]
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.
        i: Loop index

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      decoder_input = tf.slice(target_ids,
                               [0, tf.maximum(tf.cast(0, i.dtype), i - 1)],
                               [target_ids.shape[0], 1])
      self_attention_mask = tf.slice(decoder_self_attention_mask, [0, 0, i, 0],
                                     [1, 1, 1, max_decode_len])

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embeder(
          decoder_input, 1, start_pos=i, training=False)

      decoder_output = self.decoder(
          decoder_input, self_attention_mask,
          cache.get("encoder_output"), cache.get("encoder_mask"),
          cache=cache, decode_i=i, training=False)

      logits = self.embeder.linear(decoder_output)
      logits = tf.squeeze(logits, axis=[1])

      return logits

    return _symbols_to_logits_fn

  def _predict(self, target_ids, target_mask, start_token_ids,
               encoder_output, encoder_mask):
    """Beam decode output tokens and probabilities.

    Args:
      target_ids: tensor with shape [batch_size, target_length, hidden_size]
      target_mask: self-attention bias for decoder attention layer. [batch_size,
        input_length]
      start_token_ids: int32 tensor of shape [batch_size] for first decoder
        input.
      encoder_output: Continuous representation of input sequence. Float
        tensor with shape [batch_size, target_length, num_hidden_layers,
        hidden_size]
      encoder_mask: bias for encoder-decoder attention layer. [batch_size,
        input_length]

    Returns:
      A tuple of:
        `log_probs`: Log-probs of output tokens.
        `logits`: Logits of output tokens.
        `pred_ids`: Predicted output sequence.
    """
    batch_size = utils.get_shape_list(start_token_ids)[0]
    end_token_id = 1

    # One step logit function.
    symbols_to_logits_fn = self._get_symbols_to_logits_fn(target_mask)

    # Create cache storing decoder attention values for each layer.
    cache = self._init_cache(batch_size)

    if encoder_output is not None:
      # Add encoder output and attention bias to the cache.
      cache["encoder_output"] = encoder_output
      cache["encoder_mask"] = encoder_mask

    decoded_ids = decoder.left2right_decode(
        symbols_to_logits_fn,
        start_token_ids,
        cache,
        batch_size,
        self.params["max_decoder_length"],
        vocab_size=self.params["vocab_size"],
        beam_size=self.params["beam_size"],
        beam_start=5,
        beam_alpha=self.params["alpha"],
        beam_min=0,
        beam_max=-1,
        eos_id=end_token_id)

    # Get the top sequence for each batch element
    output_ids = tf.cast(decoded_ids, tf.int32, name="pred_ids")

    # Calculate log probs for given sequence if available.
    calc_ids = output_ids if target_ids is None else target_ids
    output_log_probs, output_logits, _ = self._decode(
        calc_ids, target_mask, start_token_ids,
        encoder_output, encoder_mask, training=False)

    return (output_log_probs, output_logits, output_ids)

  def _decode_and_predict(self, target_ids, encoder_output, encoder_mask,
                          training=None):
    """Decodes a sequence given the input and the encoder.

    Args:
      target_ids: tensor with shape [batch_size, target_length, hidden_size]
      encoder_output: Continuous representation of input sequence. Float
        tensor with shape [batch_size, target_length, num_hidden_layers,
        hidden_size]
      encoder_mask: bias for encoder-decoder attention layer. [batch_size,
        input_length]
      training: Boolean indicating whether the call is training or inference.

    Returns:
      A tuple of:
        `log_probs`: Log-probs of output tokens.
        `logits`: Logits of output tokens.
        `pred_ids`: Predicted output sequence.
    """
    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    start_token_ids = self._get_start_token_ids(encoder_output)

    # Create causal self-attention mask for decoder.
    target_mask = decoder.create_self_attention_mask(
        self.params["max_decoder_length"])

    predictions = {}
    if training:
      predictions = self._decode(target_ids, target_mask, start_token_ids,
                                 encoder_output, encoder_mask, training=True)
    else:
      predictions = self._predict(target_ids, target_mask, start_token_ids,
                                  encoder_output, encoder_mask)

    return predictions

  def call(self,
           input_ids,
           target_ids=None,
           training=None):
    # Run the inputs through the encoder layer to map the symbol
    # representations to continuous representations.
    encoder_output, encoder_mask = self._encode(input_ids, training=training)

    # Decode.
    predictions = self._decode_and_predict(target_ids, encoder_output,
                                           encoder_mask, training=training)

    return predictions, encoder_output
