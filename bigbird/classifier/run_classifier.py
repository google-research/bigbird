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

"""Run classification fine-tuning for BigBird."""

import os

from absl import app
from absl import logging
from bigbird.core import flags
from bigbird.core import modeling
from bigbird.core import optimization
from bigbird.core import utils
from natsort import natsorted
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft


FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string(
    "data_dir", "tfds://imdb_reviews/plain_text",
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
    "than this will be padded.")

flags.DEFINE_string(
    "substitute_newline", None,
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
    "train_batch_size", 8,
    "Local batch size for training. "
    "Total batch size will be multiplied by number gpu/tpu cores available.")

flags.DEFINE_integer(
    "eval_batch_size", 8,
    "Local batch size for eval. "
    "Total batch size will be multiplied by number gpu/tpu cores available.")

flags.DEFINE_string(
    "optimizer", "AdamWeightDecay",
    "Optimizer to use. Can be Adafactor, Adam, and AdamWeightDecay.")

flags.DEFINE_float(
    "learning_rate", 1e-5,
    "The initial learning rate for Adam.")

flags.DEFINE_integer(
    "num_train_steps", 16000,
    "Total number of training steps to perform.")

flags.DEFINE_integer(
    "num_warmup_steps", 1000,
    "Number of steps to perform linear warmup.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000,
    "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "num_labels", 2,
    "Number of ways to classify.")


def input_fn_builder(data_dir, vocab_model_file, max_encoder_length,
                     substitute_newline, is_training, tmp_dir=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "text": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(record, name_to_features)
    return example

  def _tokenize_example(example):
    text, label = example["text"], example["label"]
    tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(vocab_model_file, "rb").read())
    if substitute_newline:
      text = tf.strings.regex_replace(text, "\n", substitute_newline)
    ids = tokenizer.tokenize(text)
    ids = ids[:max_encoder_length - 2]
    # Add [CLS] (65) and [SEP] (66) special tokens.
    prefix = tf.constant([65])
    suffix = tf.constant([66])
    ids = tf.concat([prefix, ids, suffix], axis=0)
    if isinstance(ids, tf.RaggedTensor):
      ids = ids.to_tensor(0)

    # tf.Example only supports tf.int64, but the TPU is better with tf.int32.
    label = tf.cast(label, tf.int32)

    return ids, label

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
    tpu_context = params.get("context", None)
    seed = 0

    # Load dataset and handle tfds separately
    split = "train" if is_training else "test"
    if "tfds://" == data_dir[:7]:
      d = tfds.load(data_dir[7:], split=split,
                    shuffle_files=is_training,
                    data_dir=tmp_dir)
    else:
      input_files = tf.io.gfile.glob(
          os.path.join(data_dir, "{}.tfrecord*".format(split)))

      # Classification datasets are small so parallel interleaved reading
      # won't buy us much.
      d = tf.data.TFRecordDataset(input_files)
      d = d.map(_decode_record,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=is_training)

    d = d.map(_tokenize_example,
              num_parallel_calls=tf.data.experimental.AUTOTUNE,
              deterministic=is_training)

    # Tokenize and batch dataset by sentencepiece
    if is_training:
      # Classification datasets are usually small
      # and interleaving files may not be effective.
      # So to ensure different data in a multi-host setup
      # we explicitly shard the dataset by host id.
      if tpu_context:  # ensuring different data in multi-host setup
        d = d.shard(tpu_context.num_hosts, tpu_context.current_host)
        seed = tpu_context.current_host
      d = d.shuffle(buffer_size=10000, seed=seed,
                    reshuffle_each_iteration=True)
      d = d.repeat()
    d = d.padded_batch(batch_size, ([max_encoder_length], []),
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
    ids = ids[:, :max_encoder_length - 2]

    # Add [CLS] and [SEP] special tokens.
    prefix = tf.repeat(tf.constant([[65]]), batch_size, axis=0)
    suffix = tf.repeat(tf.constant([[66]]), batch_size, axis=0)
    ids = tf.concat([prefix, ids, suffix], axis=1)
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

    if isinstance(features, dict):
      if not labels and "labels" in features:
        labels = features["labels"]
      features = features["input_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(bert_config)
    headl = ClassifierLossLayer(
        bert_config["hidden_size"], bert_config["num_labels"],
        bert_config["hidden_dropout_prob"],
        utils.create_initializer(bert_config["initializer_range"]),
        name=bert_config["scope"]+"/classifier")

    _, pooled_output = model(features, training=is_training)
    total_loss, log_probs = headl(pooled_output, labels, is_training)

    tvars = tf.compat.v1.trainable_variables()
    utils.log_variables(tvars, bert_config["ckpt_var_list"])

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      learning_rate = optimization.get_linear_warmup_linear_decay_lr(
          init_lr=bert_config["learning_rate"],
          num_train_steps=bert_config["num_train_steps"],
          num_warmup_steps=bert_config["num_warmup_steps"])

      optimizer = optimization.get_optimizer(bert_config, learning_rate)

      global_step = tf.compat.v1.train.get_or_create_global_step()

      gradients = optimizer.compute_gradients(total_loss, tvars)
      train_op = optimizer.apply_gradients(gradients, global_step=global_step)

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          host_call=utils.add_scalars_to_summary(
              bert_config["output_dir"], {"learning_rate": learning_rate}))

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(loss_value, label_ids, log_probs):
        loss = tf.compat.v1.metrics.mean(values=loss_value)

        predictions = tf.argmax(log_probs, axis=-1, output_type=tf.int32)
        accuracy = tf.compat.v1.metrics.accuracy(
            labels=label_ids, predictions=predictions)
        p1, p1_op = tf.compat.v1.metrics.precision_at_k(
            labels=tf.cast(label_ids, tf.int64), predictions=log_probs, k=1)
        r1, r1_op = tf.compat.v1.metrics.recall_at_k(
            labels=tf.cast(label_ids, tf.int64), predictions=log_probs, k=1)
        f11 = tf.math.divide_no_nan(2*p1*r1, p1+r1)

        metric_dict = {
            "P@1": (p1, p1_op),
            "R@1": (r1, r1_op),
            "f1@1": (f11, tf.no_op()),
            "classification_accuracy": accuracy,
            "classification_loss": loss,
        }

        return metric_dict

      eval_metrics = (metric_fn,
                      [tf.expand_dims(total_loss, 0), labels, log_probs])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics)
    else:
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"log-probabilities": log_probs})

    return output_spec

  return model_fn


class ClassifierLossLayer(tf.keras.layers.Layer):
  """Final classifier layer with loss."""

  def __init__(self,
               hidden_size,
               num_labels,
               dropout_prob=0.0,
               initializer=None,
               use_bias=True,
               name="classifier"):
    super(ClassifierLossLayer, self).__init__(name=name)
    self.hidden_size = hidden_size
    self.num_labels = num_labels
    self.initializer = initializer
    self.dropout = tf.keras.layers.Dropout(dropout_prob)
    self.use_bias = use_bias

    with tf.compat.v1.variable_scope(name):
      self.w = tf.compat.v1.get_variable(
          name="kernel",
          shape=[self.hidden_size, self.num_labels],
          initializer=self.initializer)
      if self.use_bias:
        self.b = tf.compat.v1.get_variable(
            name="bias",
            shape=[self.num_labels],
            initializer=tf.zeros_initializer)
      else:
        self.b = None

  def call(self, input_tensor, labels=None, training=None):
    input_tensor = self.dropout(input_tensor, training)

    logits = tf.matmul(input_tensor, self.w)
    if self.use_bias:
      logits = tf.nn.bias_add(logits, self.b)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    if labels is not None:
      one_hot_labels = tf.one_hot(labels, depth=self.num_labels,
                                  dtype=tf.float32)
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
    flags.save(os.path.join(FLAGS.output_dir, "classifier.config"))

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
        max_encoder_length=FLAGS.max_encoder_length,
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
        max_encoder_length=FLAGS.max_encoder_length,
        substitute_newline=FLAGS.substitute_newline,
        tmp_dir=tmp_data_dir,
        is_training=False)

    if FLAGS.use_tpu:
      with tf.compat.v1.Session() as sess:
        eval_steps = eval_input_fn({
            "batch_size": estimator.eval_batch_size
        }).cardinality().eval(session=sess)
    else:
      eval_steps = None

    # Run evaluation for each new checkpoint.
    all_ckpts = [
        v.split(".meta")[0] for v in tf.io.gfile.glob(
            os.path.join(FLAGS.output_dir, "model.ckpt*.meta"))
    ]
    all_ckpts = natsorted(all_ckpts)
    for ckpt in all_ckpts:
      current_step = int(os.path.basename(ckpt).split("-")[1])
      output_eval_file = os.path.join(
          FLAGS.output_dir, "eval_results_{}.txt".format(current_step))
      result = estimator.evaluate(input_fn=eval_input_fn,
                                  checkpoint_path=ckpt,
                                  steps=eval_steps)

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
