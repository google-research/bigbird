# Copyright 2020 The BigBird Authors.
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

"""BigBird Decoder Layers."""

from bigbird.core import attention
from bigbird.core import beam_search
from bigbird.core import utils
import tensorflow.compat.v2 as tf


class PrenormDecoderLayer(tf.compat.v1.layers.Layer):
  """Decoder layer of a transformer in Pegasus style.

  The layer_norm is taken before self-attention.
  """

  def __init__(self,
               hidden_size=768,
               intermediate_size=3072,
               intermediate_act_fn=utils.gelu,
               attention_probs_dropout_prob=0.0,
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               num_attention_heads=12,
               use_bias=True,
               name=None):
    """Constructor of a decoder layer of a transformer in Pegasus style.

    Args:
      hidden_size: (optional) int. Size of hidden dimension.
      intermediate_size: (optional) int. Size of intermediate dimension.
      intermediate_act_fn: optional) Activation function for intermediate layer.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      hidden_dropout_prob: (optional) float. Dropout probability of the
        attention.
      initializer_range: (optional) float. Range of the weight initializer.
      num_attention_heads: (optional) int. Number of attention heads.
      use_bias: (optional) bool. Whether key/query/value uses a bias vector.
      name: The name scope of this layer.
    """
    super(PrenormDecoderLayer, self).__init__(name=name)
    self.hidden_dropout_prob = hidden_dropout_prob

    # Attention layers
    attention_head_size = hidden_size // num_attention_heads
    self.self_attn_layer = attention.MultiHeadedAttentionLayer(
        "original_full", use_bias=use_bias, name="self",
        num_attention_heads=num_attention_heads,
        size_per_head=attention_head_size,
        initializer_range=initializer_range,
        attention_probs_dropout_prob=attention_probs_dropout_prob)
    self.cross_attn_layer = attention.MultiHeadedAttentionLayer(
        "original_full", use_bias=use_bias, name="encdec",
        num_attention_heads=num_attention_heads,
        size_per_head=attention_head_size,
        initializer_range=initializer_range,
        attention_probs_dropout_prob=attention_probs_dropout_prob)

    # Dense layers
    self.self_proj_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.cross_proj_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.expand_layer = utils.Dense2dLayer(
        intermediate_size, utils.create_initializer(initializer_range),
        intermediate_act_fn, "dense")
    self.contract_layer = utils.Dense2dLayer(
        hidden_size, utils.create_initializer(initializer_range),
        None, "dense")

    # Normalization layer
    self.first_layer_norm = utils.NormLayer()
    self.second_layer_norm = utils.NormLayer()
    self.third_layer_norm = utils.NormLayer()

  @property
  def trainable_weights(self):
    tvar_list = (self.self_attn_layer.trainable_weights +
                 self.cross_attn_layer.trainable_weights +
                 self.self_proj_layer.trainable_weights +
                 self.cross_proj_layer.trainable_weights +
                 self.expand_layer.trainable_weights +
                 self.contract_layer.trainable_weights +
                 self.first_layer_norm.trainable_weights +
                 self.second_layer_norm.trainable_weights +
                 self.third_layer_norm.trainable_weights)
    self._trainable_weights = list({v.name: v for v in tvar_list}.values())
    return self._trainable_weights

  def call(self,
           layer_input,
           encoder_outputs,
           self_attention_mask,
           attention_mask,
           cache=None,
           decode_i=None,
           training=None):
    """Implements a decoder layer of a transformer in Pegasus style.

    The layer_norm is taken after self-attention.

    Args:
      layer_input: float Tensor of shape [batch_size, seq_length, hidden_size].
      encoder_outputs: tensors with shape [batch_size, input_length,
          num_hidden_layers, hidden_size]
      self_attention_mask: bias for decoder self-attention layer. [1, 1,
        target_length, target_length]
      attention_mask: bias for encoder-decoder attention layer. [batch_size, 1,
        1, input_length]
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head],
             "v": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head]}
      decode_i: (Used during prediction) current location of decoding
      training: Boolean indicating whether the call is training or inference.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size].

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
      NotImplementedError: For unknown attention type.
    """
    with tf.compat.v1.variable_scope("attention"):
      with tf.compat.v1.variable_scope("self") as sc:
        normalized_layer_input = self.first_layer_norm(layer_input)
        self_attention_output = self.self_attn_layer(
            normalized_layer_input, normalized_layer_input, self_attention_mask,
            cache=cache, decode_i=decode_i, training=training, scope=sc)

      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      with tf.compat.v1.variable_scope("output"):
        self_attention_output = self.self_proj_layer(self_attention_output)
        self_attention_output = utils.dropout(self_attention_output,
                                              self.hidden_dropout_prob,
                                              training)
        self_attention_output = self_attention_output + layer_input

      with tf.compat.v1.variable_scope("encdec") as sc:
        normalized_self_attention_output = self.second_layer_norm(
            self_attention_output)
        attention_output = self.cross_attn_layer(
            normalized_self_attention_output, encoder_outputs, attention_mask,
            training=training, scope=sc)

      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      with tf.compat.v1.variable_scope("encdec_output"):
        attention_output = self.cross_proj_layer(attention_output)
        attention_output = utils.dropout(attention_output,
                                         self.hidden_dropout_prob,
                                         training)
        attention_output = attention_output + self_attention_output

    # The activation is only applied to the "intermediate" hidden layer.
    with tf.compat.v1.variable_scope("intermediate"):
      normalized_attention_output = self.third_layer_norm(attention_output)
      intermediate_output = self.expand_layer(normalized_attention_output)

    # Down-project back to `hidden_size` then add the residual.
    with tf.compat.v1.variable_scope("output"):
      layer_output = self.contract_layer(intermediate_output)
      layer_output = utils.dropout(layer_output,
                                   self.hidden_dropout_prob,
                                   training)
      layer_output = layer_output + attention_output
    return layer_output


class PostnormDecoderLayer(tf.compat.v1.layers.Layer):
  """Decoder layer of a transformer in BERT style.

  The layer_norm is taken before self-attention.
  """

  def __init__(self,
               hidden_size=768,
               intermediate_size=3072,
               intermediate_act_fn=utils.gelu,
               attention_probs_dropout_prob=0.0,
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               num_attention_heads=12,
               use_bias=True,
               name=None):
    """Constructor of a decoder layer of a transformer in BERT style.

    Args:
      hidden_size: (optional) int. Size of hidden dimension.
      intermediate_size: (optional) int. Size of intermediate dimension.
      intermediate_act_fn: optional) Activation function for intermediate layer.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      hidden_dropout_prob: (optional) float. Dropout probability of the
        attention.
      initializer_range: (optional) float. Range of the weight initializer.
      num_attention_heads: (optional) int. Number of attention heads.
      use_bias: (optional) bool. Whether key/query/value uses a bias vector.
      name: The name scope of this layer.
    """
    super(PostnormDecoderLayer, self).__init__(name=name)
    self.hidden_dropout_prob = hidden_dropout_prob

    # Attention layers
    attention_head_size = hidden_size // num_attention_heads
    self.self_attn_layer = attention.MultiHeadedAttentionLayer(
        "original_full", use_bias=use_bias, name="self",
        num_attention_heads=num_attention_heads,
        size_per_head=attention_head_size,
        initializer_range=initializer_range,
        attention_probs_dropout_prob=attention_probs_dropout_prob)
    self.cross_attn_layer = attention.MultiHeadedAttentionLayer(
        "original_full", use_bias=use_bias, name="encdec",
        num_attention_heads=num_attention_heads,
        size_per_head=attention_head_size,
        initializer_range=initializer_range,
        attention_probs_dropout_prob=attention_probs_dropout_prob)

    # Dense layers
    self.self_proj_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.cross_proj_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.expand_layer = utils.Dense2dLayer(
        intermediate_size, utils.create_initializer(initializer_range),
        intermediate_act_fn, "dense")
    self.contract_layer = utils.Dense2dLayer(
        hidden_size, utils.create_initializer(initializer_range),
        None, "dense")

    # Normalization layer
    self.first_layer_norm = utils.NormLayer()
    self.second_layer_norm = utils.NormLayer()
    self.third_layer_norm = utils.NormLayer()

  @property
  def trainable_weights(self):
    tvar_list = (self.self_attn_layer.trainable_weights +
                 self.cross_attn_layer.trainable_weights +
                 self.self_proj_layer.trainable_weights +
                 self.cross_proj_layer.trainable_weights +
                 self.expand_layer.trainable_weights +
                 self.contract_layer.trainable_weights +
                 self.first_layer_norm.trainable_weights +
                 self.second_layer_norm.trainable_weights +
                 self.third_layer_norm.trainable_weights)
    self._trainable_weights = list({v.name: v for v in tvar_list}.values())
    return self._trainable_weights

  def call(self,
           layer_input,
           encoder_outputs,
           self_attention_mask,
           attention_mask,
           cache=None,
           decode_i=None,
           training=None):
    """Implements a decoder layer of a transformer in BERT style.

    The layer_norm is taken after self-attention.

    Args:
      layer_input: float Tensor of shape [batch_size, seq_length, hidden_size].
      encoder_outputs: tensors with shape [batch_size, input_length,
          num_hidden_layers, hidden_size]
      self_attention_mask: bias for decoder self-attention layer. [1, 1,
        target_length, target_length]
      attention_mask: bias for encoder-decoder attention layer. [batch_size, 1,
        1, input_length]
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head],
             "v": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head]}
      decode_i: (Used during prediction) current location of decoding
      training: Boolean indicating whether the call is training or inference.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size].

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
      NotImplementedError: For unknown attention type.
    """
    with tf.compat.v1.variable_scope("attention"):
      with tf.compat.v1.variable_scope("self") as sc:
        self_attention_output = self.self_attn_layer(
            layer_input, layer_input, self_attention_mask,
            cache=cache, decode_i=decode_i, training=training, scope=sc)

      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      with tf.compat.v1.variable_scope("output"):
        self_attention_output = self.self_proj_layer(self_attention_output)
        self_attention_output = utils.dropout(self_attention_output,
                                              self.hidden_dropout_prob,
                                              training)
        self_attention_output = self.first_layer_norm(
            self_attention_output + layer_input)

      with tf.compat.v1.variable_scope("encdec") as sc:
        attention_output = self.cross_attn_layer(
            self_attention_output, encoder_outputs, attention_mask,
            training=training, scope=sc)

      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      with tf.compat.v1.variable_scope("encdec_output"):
        attention_output = self.cross_proj_layer(attention_output)
        attention_output = utils.dropout(attention_output,
                                         self.hidden_dropout_prob,
                                         training)
        attention_output = self.second_layer_norm(
            attention_output + self_attention_output)

    # The activation is only applied to the "intermediate" hidden layer.
    with tf.compat.v1.variable_scope("intermediate"):
      intermediate_output = self.expand_layer(attention_output)

    # Down-project back to `hidden_size` then add the residual.
    with tf.compat.v1.variable_scope("output"):
      layer_output = self.contract_layer(intermediate_output)
      layer_output = utils.dropout(layer_output,
                                   self.hidden_dropout_prob,
                                   training)
      layer_output = self.third_layer_norm(layer_output + attention_output)
    return layer_output


class DecoderStack(tf.compat.v1.layers.Layer):
  """Transformer decoder stack."""

  def __init__(self, params):
    if params["couple_encoder_decoder"]:
      name = "encoder"
      with tf.compat.v1.variable_scope(
          name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
        super(DecoderStack, self).__init__(name=name, _scope=scope)
    else:
      name = "decoder"
      super(DecoderStack, self).__init__(name=name)

    self.params = params

    if params["norm_type"] == "prenorm":
      decoder_class = PrenormDecoderLayer
    elif params["norm_type"] == "postnorm":
      decoder_class = PostnormDecoderLayer
    else:
      raise NotImplementedError(
          "Norm type {} is not implemented".format(params["norm_type"]))

    if self.params.get("num_decoder_layers", None) is not None:
      num_hidden_layers = self.params["num_decoder_layers"]
    else:
      num_hidden_layers = self.params["num_hidden_layers"]

    # Decoder layers
    self.decoder_layers = [
        decoder_class(  # pylint: disable=g-complex-comprehension
            self.params["hidden_size"],
            self.params["intermediate_size"],
            utils.get_activation(self.params["hidden_act"]),
            self.params["attention_probs_dropout_prob"],
            self.params["hidden_dropout_prob"],
            self.params["initializer_range"],
            self.params["num_attention_heads"],
            self.params["use_bias"],
            name="layer_%d" % layer_idx)
        for layer_idx in range(num_hidden_layers)
    ]

    # Normalization layer
    self.layer_norm = utils.NormLayer()

  @property
  def trainable_weights(self):
    tvar_list = sum(
        [layer.trainable_weights for layer in self.decoder_layers],
        []) +  self.layer_norm.trainable_weights
    self._trainable_weights = list({v.name: v for v in tvar_list}.values())
    return self._trainable_weights

  def call(self,
           decoder_inputs,
           self_attention_mask,
           encoder_outputs,
           encoder_mask,
           cache=None,
           decode_i=None,
           training=None):
    """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: tensor with shape
        [batch_size, target_length, hidden_size]
      self_attention_mask: bias for decoder self-attention layer. [1, 1,
        target_length, target_length]
      encoder_outputs: tensors with shape [batch_size, input_length,
        hidden_size]
      encoder_mask: bias for encoder-decoder attention layer. [batch_size,
        input_length]
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head],
             "v": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head]}
      decode_i: (Used during prediction) current location of decoding.
      training: Boolean indicating whether the call is training or inference.

    Returns:
      Output of decoder layer stack. A float32 tensor with shape [batch_size,
        target_length, hidden_size]
    """
    # Expand encoder mask to broadcast over num heads and from_seq axis
    attention_mask = tf.expand_dims(tf.expand_dims(encoder_mask, 1), 1)

    # if self.params["use_gradient_checkpointing"]::
    #   decoder_layer = recompute_gradient(decoder_layer)

    if self.params["norm_type"] == "postnorm":
      decoder_inputs = self.layer_norm(decoder_inputs)

    layer_output = decoder_inputs
    for layer in self.decoder_layers:
      layer_cache = cache[layer.name] if cache is not None else None
      layer_output = layer(
          layer_output, encoder_outputs, self_attention_mask, attention_mask,
          layer_cache, decode_i, training)

    if self.params["norm_type"] == "prenorm":
      layer_output = self.layer_norm(layer_output)

    return layer_output


def create_self_attention_mask(length):
  with tf.name_scope("decoder_self_attention_mask"):
    valid_locs = tf.linalg.band_part(tf.ones([length, length]), -1, 0)
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
  return valid_locs


def inplace_update_i(inp_tensor, updates, i):
  """Inplace update a tensor. B: batch_size, L: tensor length."""
  batch_size = inp_tensor.shape[0]
  indices = tf.stack([
      tf.range(batch_size, dtype=tf.int32),
      tf.fill([batch_size], tf.cast(i, tf.int32))
  ], axis=-1)
  return tf.tensor_scatter_nd_update(inp_tensor, indices, updates)


# pylint: disable=invalid-name
def left2right_decode(symbols_to_logits_fn,
                      start_symbols,
                      context_BxU_dict,
                      batch_size,
                      max_decode_len,
                      vocab_size,
                      beam_size=1,
                      beam_start=5,
                      beam_alpha=0.6,
                      beam_min=0,
                      beam_max=-1,
                      eos_id=1):
  """left to right decode.

  Notations:
    B: batch_size, V: vocab_size, T: decode_len, U: undefined dimensions

  Args:
    symbols_to_logits_fn: logits = fn(decodes, context, i). Shoud take
      [batch_size, decoded_ids] and return [batch_size, vocab_size].
    start_symbols: starting ids [batch_size]
    context_BxU_dict: dict of Tensors.
    batch_size: int, decode batch size.
    max_decode_len: int, maximum number of steps to decode.
    vocab_size: int, output vocab size.
    beam_size: Number of beams to decode.
    beam_start: start length for scaling, default to 5.
    beam_alpha: Length penalty for decoding. Should be between 0 (shorter) and 1
      (longer), default to 0.6.
    beam_min: Minimum beam search lengths.
    beam_max: Maximum beam search lengths. Set -1 to use unlimited.
    eos_id: end of token id, default to 1.

  Returns:
    decodes: Tensor[batch, decode_len]
  """
  dtype = tf.int32
  start_symbols = tf.expand_dims(start_symbols, 1)
  # When beam_size=1, beam_search does not behave exactly like greedy.
  # This is due to using 2 * beam_size in grow_topk, and keep the top beam_size
  # ones that haven't reached EOS into alive.
  # In this case, alpha value for length penalty will take effect.
  if beam_size == 1:

    def decode_loop(i, decodes_BxT, cache_BxU_dict):
      logits_BxV = symbols_to_logits_fn(decodes_BxT, cache_BxU_dict, i)
      decodes_BxT = inplace_update_i(
          decodes_BxT, tf.argmax(logits_BxV, -1, output_type=tf.int32), i)
      return i + 1, decodes_BxT, cache_BxU_dict

    def loop_cond(i, decodes_BxT, unused_cache_BxU_dict):
      finished_B = tf.reduce_any(tf.equal(decodes_BxT, eos_id), axis=1)
      return tf.logical_and(i < max_decode_len,
                            tf.logical_not(tf.reduce_all(finished_B)))

    init_dec_BxT = tf.concat([tf.cast(start_symbols, dtype=dtype),
                              tf.zeros([batch_size, max_decode_len-1],
                                       dtype=dtype)], axis=1)
    _, decodes, _ = tf.while_loop(
        loop_cond, decode_loop,
        [tf.constant(0, dtype=dtype), init_dec_BxT, context_BxU_dict])
    return decodes

  else:

    def symbols_to_logits_fn_with_sampling(decodes_BxT, states_BxU_dict, i):
      logits_BxV = symbols_to_logits_fn(decodes_BxT, states_BxU_dict, i)
      return logits_BxV, states_BxU_dict

    length_norm_fn = beam_search.length_normalization(beam_start, beam_alpha,
                                                      beam_min, beam_max, -1e3)

    init_dec_BxT = tf.concat([tf.cast(start_symbols, dtype=tf.int32),
                              tf.zeros([batch_size, max_decode_len-1],
                                       dtype=tf.int32)], axis=1)

    beams, _ = beam_search.beam_search(
        symbols_to_logits_fn_with_sampling,
        init_dec_BxT,
        context_BxU_dict, vocab_size, beam_size, length_norm_fn, eos_id)
    return beams[:, 0, :]
