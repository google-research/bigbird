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

"""Run masked LM/next sentence pre-training for BigBird."""

import os
import time

from absl import app
from absl import logging
from bigbird.core import flags
from bigbird.core import modeling
from bigbird.core import optimization
from bigbird.core import utils
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft

import sentencepiece as spm


FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string(
    "data_dir", "tfds://wiki40b/en",
    "The input data dir. Should contain the TFRecord files. "
    "Can be TF Dataset with prefix tfds://")

flags.DEFINE_string(
    "output_dir", "/tmp/bigb",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BigBird model).")

flags.DEFINE_integer(
    "max_encoder_length", 512,
    "The maximum total input sequence length after SentencePiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 75,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_float(
    "masked_lm_prob", 0.15,
    "Masked LM probability.")

flags.DEFINE_string(
    "substitute_newline", " ",
    "Replace newline charachter from text with supplied string.")

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training.")

flags.DEFINE_bool(
    "do_eval", False,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_export", False,
    "Whether to export the model as TF SavedModel.")

flags.DEFINE_integer(
    "train_batch_size", 4,
    "Local batch size for training. "
    "Total batch size will be multiplied by number gpu/tpu cores available.")

flags.DEFINE_integer(
    "eval_batch_size", 4,
    "Local batch size for eval. "
    "Total batch size will be multiplied by number gpu/tpu cores available.")

flags.DEFINE_string(
    "optimizer", "AdamWeightDecay",
    "Optimizer to use. Can be Adafactor, Adam, and AdamWeightDecay.")

flags.DEFINE_float(
    "learning_rate", 1e-4,
    "The initial learning rate for Adam.")

flags.DEFINE_integer(
    "num_train_steps", 100000,
    "Total number of training steps to perform.")

flags.DEFINE_integer(
    "num_warmup_steps", 10000,
    "Number of steps to perform linear warmup.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000,
    "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "max_eval_steps", 100,
    "Maximum number of eval steps.")

flags.DEFINE_bool(
    "preprocessed_data", False,
    "Whether TFRecord data is already tokenized and masked.")

flags.DEFINE_bool(
    "use_nsp", False,
    "Whether to use next sentence prediction loss.")


def input_fn_builder(data_dir, vocab_model_file, masked_lm_prob,
                     max_encoder_length, max_predictions_per_seq,
                     preprocessed_data, substitute_newline, is_training,
                     tmp_dir=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  sp_model = spm.SentencePieceProcessor()
  sp_proto = tf.io.gfile.GFile(vocab_model_file, "rb").read()
  sp_model.LoadFromSerializedProto(sp_proto)
  vocab_size = sp_model.GetPieceSize()
  word_start_subtoken = np.array(
      [sp_model.IdToPiece(i)[0] == "▁" for i in range(vocab_size)])

  feature_shapes = {
      "input_ids": [max_encoder_length],
      "segment_ids": [max_encoder_length],
      "masked_lm_positions": [max_predictions_per_seq],
      "masked_lm_ids": [max_predictions_per_seq],
      "masked_lm_weights": [max_predictions_per_seq],
      "next_sentence_labels": [1]
  }

  def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "input_ids":
            tf.io.FixedLenFeature([max_encoder_length], tf.int64),
        "segment_ids":
            tf.io.FixedLenFeature([max_encoder_length], tf.int64),
        "masked_lm_positions":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.io.FixedLenFeature([1], tf.int64),
    }
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def do_masking(example):
    text = example["text"]
    tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(vocab_model_file, "rb").read())
    if substitute_newline:
      text = tf.strings.regex_replace(text, "\n", substitute_newline)
    subtokens = tokenizer.tokenize(text)
    (subtokens, masked_lm_positions, masked_lm_ids,
     masked_lm_weights) = tf.compat.v1.py_func(
         numpy_masking, [subtokens], [tf.int32, tf.int32, tf.int32, tf.float32],
         stateful=False)
    features = {
        "input_ids": subtokens,
        "segment_ids": tf.zeros_like(subtokens),
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_ids": masked_lm_ids,
        "masked_lm_weights": masked_lm_weights,
        "next_sentence_labels": tf.zeros([1], dtype=tf.int64),
    }
    return features

  def numpy_masking(subtokens):
    # Find a random span in text
    end_pos = max_encoder_length - 2 + np.random.randint(
        max(1, len(subtokens) - max_encoder_length - 2))
    start_pos = max(0, end_pos - max_encoder_length + 2)
    subtokens = subtokens[start_pos:end_pos]

    # The start might be inside a word so fix it
    # such that span always starts at a word
    word_begin_mark = word_start_subtoken[subtokens]
    word_begins_pos = np.flatnonzero(word_begin_mark).astype(np.int32)
    if word_begins_pos.size == 0:
      # if no word boundary present, we do not do whole word masking
      # and we fall back to random masking.
      word_begins_pos = np.arange(len(subtokens), dtype=np.int32)
      word_begin_mark = np.logical_not(word_begin_mark)
      print(subtokens, start_pos, end_pos, word_begin_mark)
    correct_start_pos = word_begins_pos[0]
    subtokens = subtokens[correct_start_pos:]
    word_begin_mark = word_begin_mark[correct_start_pos:]
    word_begins_pos = word_begins_pos - correct_start_pos
    num_tokens = len(subtokens)

    # @e want to do whole word masking so split by word boundary
    words = np.split(np.arange(num_tokens, dtype=np.int32), word_begins_pos)[1:]
    assert len(words) == len(word_begins_pos)

    # Decide elements to mask
    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(len(word_begins_pos) * masked_lm_prob))))
    masked_lm_positions = np.concatenate(np.random.choice(
        np.array([[]] + words, dtype=np.object)[1:],
        num_to_predict, replace=False), 0)
    # but this might have excess subtokens than max_predictions_per_seq
    if len(masked_lm_positions) > max_predictions_per_seq:
      masked_lm_positions = masked_lm_positions[:max_predictions_per_seq+1]
      # however last word can cross word boundaries, remove crossing words
      truncate_masking_at = np.flatnonzero(
          word_begin_mark[masked_lm_positions])[-1]
      masked_lm_positions = masked_lm_positions[:truncate_masking_at]

    # sort masking positions
    masked_lm_positions = np.sort(masked_lm_positions)
    masked_lm_ids = subtokens[masked_lm_positions]

    # replance input token with [MASK] 80%, random 10%, or leave it as it is.
    randomness = np.random.rand(len(masked_lm_positions))
    mask_index = masked_lm_positions[randomness < 0.8]
    random_index = masked_lm_positions[randomness > 0.9]

    subtokens[mask_index] = 67  # id of masked token
    subtokens[random_index] = np.random.randint(  # ignore special tokens
        101, vocab_size, len(random_index), dtype=np.int32)

    # add [CLS] (65) and [SEP] (66) tokens
    subtokens = np.concatenate([
        np.array([65], dtype=np.int32), subtokens,
        np.array([66], dtype=np.int32)
    ])

    # pad everything to correct shape
    pad_inp = max_encoder_length - num_tokens - 2
    subtokens = np.pad(subtokens, [0, pad_inp], "constant")

    pad_out = max_predictions_per_seq - len(masked_lm_positions)
    masked_lm_weights = np.pad(
        np.ones_like(masked_lm_positions, dtype=np.float32),
        [0, pad_out], "constant")
    masked_lm_positions = np.pad(
        masked_lm_positions + 1, [0, pad_out], "constant")
    masked_lm_ids = np.pad(masked_lm_ids, [0, pad_out], "constant")

    return subtokens, masked_lm_positions, masked_lm_ids, masked_lm_weights

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # Load dataset and handle tfds separately
    split = "train" if is_training else "test"
    if "tfds://" == data_dir[:7]:
      d = tfds.load(data_dir[7:], split=split,
                    shuffle_files=is_training,
                    data_dir=tmp_dir)
    else:
      input_files = tf.io.gfile.glob(
          os.path.join(data_dir, "{}.tfrecord*".format(split)))

      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.shuffle(buffer_size=len(input_files))

        # Non deterministic mode means that the interleaving is not exact.
        # This adds even more randomness to the training pipeline.
        d = d.interleave(tf.data.TFRecordDataset,
                         deterministic=False,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
      else:
        d = tf.data.TFRecordDataset(input_files)

    if preprocessed_data:
      d = d.map(_decode_record,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      d = d.map(do_masking,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
      d = d.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
      d = d.repeat()

    d = d.padded_batch(batch_size, feature_shapes,
                       drop_remainder=True)  # For static shape
    return d

  return input_fn


def serving_input_fn_builder(batch_size, max_encoder_length,
                             vocab_model_file, substitute_newline):
  """Creates an `input_fn` closure for exported SavedModel."""
  def dynamic_padding(inp, min_size):
    pad_size = tf.maximum(min_size - tf.shape(inp)[1], 0)
    paddings = [[0, 0], [0, pad_size]]
    return tf.pad(inp, paddings)

  def input_fn():
    # text input
    text = tf.compat.v1.placeholder(tf.string, [batch_size], name="input_text")

    # text tokenize
    tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(vocab_model_file, "rb").read())
    if substitute_newline:
      text = tf.strings.regex_replace(text, "\n", substitute_newline)
    ids = tokenizer.tokenize(text)
    if isinstance(ids, tf.RaggedTensor):
      ids = ids.to_tensor(0)

    # text padding: Pad only if necessary and reshape properly
    padded_ids = dynamic_padding(ids, max_encoder_length)
    ids = tf.slice(padded_ids, [0, 0], [batch_size, max_encoder_length])

    receiver_tensors = {"input": text}
    features = {"input_ids": tf.cast(ids, tf.int32, name="input_ids")}

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=receiver_tensors)

  return input_fn


def model_fn_builder(bert_config):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(bert_config)
    masked_lm = MaskedLMLayer(
        bert_config["hidden_size"], bert_config["vocab_size"], model.embeder,
        initializer=utils.create_initializer(bert_config["initializer_range"]),
        activation_fn=utils.get_activation(bert_config["hidden_act"]))
    next_sentence = NSPLayer(
        bert_config["hidden_size"],
        initializer=utils.create_initializer(bert_config["initializer_range"]))

    sequence_output, pooled_output = model(
        features["input_ids"], training=is_training,
        token_type_ids=features.get("segment_ids"))

    masked_lm_loss, masked_lm_log_probs = masked_lm(
        sequence_output,
        label_ids=features.get("masked_lm_ids"),
        label_weights=features.get("masked_lm_weights"),
        masked_lm_positions=features.get("masked_lm_positions"))

    next_sentence_loss, next_sentence_log_probs = next_sentence(
        pooled_output, features.get("next_sentence_labels"))

    total_loss = masked_lm_loss
    if bert_config["use_nsp"]:
      total_loss += next_sentence_loss

    tvars = tf.compat.v1.trainable_variables()
    utils.log_variables(tvars, bert_config["ckpt_var_list"])

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      learning_rate = optimization.get_linear_warmup_linear_decay_lr(
          init_lr=bert_config["learning_rate"],
          num_train_steps=bert_config["num_train_steps"],
          num_warmup_steps=bert_config["num_warmup_steps"])

      optimizer = optimization.get_optimizer(bert_config, learning_rate)

      global_step = tf.compat.v1.train.get_global_step()

      gradients = optimizer.compute_gradients(total_loss, tvars)
      train_op = optimizer.apply_gradients(gradients, global_step=global_step)

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          host_call=utils.add_scalars_to_summary(
              bert_config["output_dir"], {"learning_rate": learning_rate}))

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_loss_value, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_loss_value,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.compat.v1.metrics.mean(
            values=masked_lm_loss_value)

        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.compat.v1.metrics.mean(
            values=next_sentence_loss_value)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_loss, masked_lm_log_probs, features["masked_lm_ids"],
          features["masked_lm_weights"], next_sentence_loss,
          next_sentence_log_probs, features["next_sentence_labels"]
      ])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics)
    else:

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "log-probabilities": masked_lm_log_probs,
              "seq-embeddings": sequence_output
          })

    return output_spec

  return model_fn


class MaskedLMLayer(tf.keras.layers.Layer):
  """Get loss and log probs for the masked LM."""

  def __init__(self,
               hidden_size,
               vocab_size,
               embeder,
               initializer=None,
               activation_fn=None,
               name="cls/predictions"):
    super(MaskedLMLayer, self).__init__(name=name)
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.embeder = embeder

    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    self.extra_layer = utils.Dense2dLayer(
        hidden_size, hidden_size, initializer,
        activation_fn, "transform")
    self.norm_layer = utils.NormLayer(hidden_size, name="transform")

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    self.output_bias = tf.compat.v1.get_variable(
        name+"/output_bias",
        shape=[vocab_size],
        initializer=tf.zeros_initializer())

  @property
  def trainable_weights(self):
    self._trainable_weights = (self.extra_layer.trainable_weights +
                               self.norm_layer.trainable_weights +
                               [self.output_bias])
    return self._trainable_weights

  def call(self, input_tensor,
           label_ids=None,
           label_weights=None,
           masked_lm_positions=None):
    if masked_lm_positions is not None:
      input_tensor = tf.gather(input_tensor, masked_lm_positions, batch_dims=1)

    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    input_tensor = self.extra_layer(input_tensor)
    input_tensor = self.norm_layer(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    logits = self.embeder.linear(input_tensor)
    logits = tf.nn.bias_add(logits, self.output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    if label_ids is not None:
      one_hot_labels = tf.one_hot(
          label_ids, depth=self.vocab_size, dtype=tf.float32)

      # The `positions` tensor might be zero-padded (if the sequence is too
      # short to have the maximum number of predictions). The `label_weights`
      # tensor has a value of 1.0 for every real prediction and 0.0 for the
      # padding predictions.
      per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=-1)
      numerator = tf.reduce_sum(label_weights * per_example_loss)
      denominator = tf.reduce_sum(label_weights) + 1e-5
      loss = numerator / denominator
    else:
      loss = tf.constant(0.0)

    return loss, log_probs


class NSPLayer(tf.keras.layers.Layer):
  """Get loss and log probs for the next sentence prediction."""

  def __init__(self,
               hidden_size,
               initializer=None,
               name="cls/seq_relationship"):
    super(NSPLayer, self).__init__(name=name)
    self.hidden_size = hidden_size

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.compat.v1.variable_scope(name):
      self.output_weights = tf.compat.v1.get_variable(
          "output_weights",
          shape=[2, hidden_size],
          initializer=initializer)
      self._trainable_weights.append(self.output_weights)
      self.output_bias = tf.compat.v1.get_variable(
          "output_bias", shape=[2], initializer=tf.zeros_initializer())
      self._trainable_weights.append(self.output_bias)

  def call(self, input_tensor, next_sentence_labels=None):
    logits = tf.matmul(input_tensor, self.output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, self.output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    if next_sentence_labels is not None:
      labels = tf.reshape(next_sentence_labels, [-1])
      one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)
    else:
      loss = tf.constant(0.0)
    return loss, log_probs


def main(_):

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_export:
    raise ValueError(
        "At least one of `do_train`, `do_eval` must be True.")

  bert_config = flags.as_dictionary()

  if FLAGS.max_encoder_length > bert_config["max_position_embeddings"]:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_encoder_length, bert_config["max_position_embeddings"]))

  tf.io.gfile.makedirs(FLAGS.output_dir)
  if FLAGS.do_train:
    flags.save(os.path.join(FLAGS.output_dir, "pretrain.config"))

  model_fn = model_fn_builder(bert_config)
  estimator = utils.get_estimator(bert_config, model_fn)
  tmp_data_dir = os.path.join(FLAGS.output_dir, "tfds")

  if FLAGS.do_train:
    logging.info("***** Running training *****")
    logging.info("  Batch size = %d", estimator.train_batch_size)
    logging.info("  Num steps = %d", FLAGS.num_train_steps)
    train_input_fn = input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        masked_lm_prob=FLAGS.masked_lm_prob,
        max_encoder_length=FLAGS.max_encoder_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        preprocessed_data=FLAGS.preprocessed_data,
        substitute_newline=FLAGS.substitute_newline,
        tmp_dir=tmp_data_dir,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    logging.info("***** Running evaluation *****")
    logging.info("  Batch size = %d", estimator.eval_batch_size)

    eval_input_fn = input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        masked_lm_prob=FLAGS.masked_lm_prob,
        max_encoder_length=FLAGS.max_encoder_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        preprocessed_data=FLAGS.preprocessed_data,
        substitute_newline=FLAGS.substitute_newline,
        tmp_dir=tmp_data_dir,
        is_training=False)

    # Run continuous evaluation for latest checkpoint as training progresses.
    last_evaluated = None
    while True:
      latest = tf.train.latest_checkpoint(FLAGS.output_dir)
      if latest == last_evaluated:
        if not latest:
          logging.info("No checkpoints found yet.")
        else:
          logging.info("Latest checkpoint %s already evaluated.", latest)
        time.sleep(300)
        continue
      else:
        logging.info("Evaluating check point %s", latest)
        last_evaluated = latest

        current_step = int(os.path.basename(latest).split("-")[1])
        output_eval_file = os.path.join(
            FLAGS.output_dir, "eval_results_{}.txt".format(current_step))
        result = estimator.evaluate(input_fn=eval_input_fn,
                                    steps=FLAGS.max_eval_steps,
                                    checkpoint_path=latest)

        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
          logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_export:
    logging.info("***** Running export *****")

    serving_input_fn = serving_input_fn_builder(
        batch_size=FLAGS.eval_batch_size,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length,
        substitute_newline=FLAGS.substitute_newline)

    estimator.export_saved_model(
        os.path.join(FLAGS.output_dir, "export"), serving_input_fn)


if __name__ == "__main__":
  tf.compat.v1.disable_v2_behavior()
  tf.compat.v1.enable_resource_variables()
  app.run(main)
