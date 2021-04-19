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

"""Run summarization fine-tuning for BigBird.."""

import os
import time

from absl import app
from absl import logging
from bigbird.core import flags
from bigbird.core import modeling
from bigbird.core import optimization
from bigbird.core import utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft


from rouge_score import rouge_scorer

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string(
    "data_dir", "tfds://scientific_papers/pubmed",
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
    "max_encoder_length", 128,
    "The maximum total input sequence length after SentencePiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_decoder_length", 128,
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
    "optimizer", "Adafactor",
    "Optimizer to use. Can be Adafactor, Adam, and AdamWeightDecay.")

flags.DEFINE_float(
    "learning_rate", 0.32,
    "The initial learning rate for Adam.")

flags.DEFINE_integer(
    "num_train_steps", 1000,
    "Total number of training steps to perform.")

flags.DEFINE_integer(
    "num_warmup_steps", 100,
    "Number of steps to perform linear warmup.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 2000,
    "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "max_eval_steps", 100,
    "Maximum number of eval steps.")

flags.DEFINE_bool(
    "couple_encoder_decoder", False,
    "Whether to tie encoder and decoder weights.")

flags.DEFINE_integer(
    "beam_size", 5,
    "Beam size for decoding.")

flags.DEFINE_float(
    "alpha", 0.8,
    "Strength of length normalization for beam search.")

flags.DEFINE_float(
    "label_smoothing", 0.1,
    "Label smoothing for prediction cross entropy loss.")


def input_fn_builder(data_dir, vocab_model_file, max_encoder_length,
                     max_decoder_length, substitute_newline, is_training,
                     tmp_dir=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "document": tf.io.FixedLenFeature([], tf.string),
        "summary": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(record, name_to_features)
    return example["document"], example["summary"]

  def _tokenize_example(document, summary):
    tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(vocab_model_file, "rb").read())
    if substitute_newline:
      document = tf.strings.regex_replace(document, "\n", substitute_newline)
    # Remove space before special tokens.
    document = tf.strings.regex_replace(document, r" ([<\[]\S+[>\]])", b"\\1")
    document_ids = tokenizer.tokenize(document)
    if isinstance(document_ids, tf.RaggedTensor):
      document_ids = document_ids.to_tensor(0)
    document_ids = document_ids[:max_encoder_length]

    # Remove newline optionally
    if substitute_newline:
      summary = tf.strings.regex_replace(summary, "\n", substitute_newline)
    # Remove space before special tokens.
    summary = tf.strings.regex_replace(summary, r" ([<\[]\S+[>\]])", b"\\1")
    summary_ids = tokenizer.tokenize(summary)
    # Add [EOS] (1) special tokens.
    suffix = tf.constant([1])
    summary_ids = tf.concat([summary_ids, suffix], axis=0)
    if isinstance(summary_ids, tf.RaggedTensor):
      summary_ids = summary_ids.to_tensor(0)
    summary_ids = summary_ids[:max_decoder_length]

    return document_ids, summary_ids

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # Load dataset and handle tfds separately
    split = "train" if is_training else "validation"
    if "tfds://" == data_dir[:7]:
      d = tfds.load(data_dir[7:], split=split, data_dir=tmp_dir,
                    shuffle_files=is_training, as_supervised=True)
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

      d = d.map(_decode_record,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=is_training)

    d = d.map(_tokenize_example,
              num_parallel_calls=tf.data.experimental.AUTOTUNE,
              deterministic=is_training)

    if is_training:
      d = d.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
      d = d.repeat()
    d = d.padded_batch(batch_size, ([max_encoder_length], [max_decoder_length]),
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
    # Remove space before special tokens.
    text = tf.strings.regex_replace(text, r" ([<\[]\S+[>\]])", b"\\1")
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


def model_fn_builder(transformer_config):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    if isinstance(features, dict):
      if not labels and "target_ids" in features:
        labels = features["target_ids"]
      features = features["input_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.TransformerModel(transformer_config)
    (llh, logits, pred_ids), _ = model(features, target_ids=labels,
                                       training=is_training)

    total_loss = padded_cross_entropy_loss(
        logits, labels,
        transformer_config["label_smoothing"],
        transformer_config["vocab_size"])

    tvars = tf.compat.v1.trainable_variables()
    utils.log_variables(tvars, transformer_config["ckpt_var_list"])

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      learning_rate = optimization.get_linear_warmup_rsqrt_decay_lr(
          init_lr=transformer_config["learning_rate"],
          hidden_size=transformer_config["hidden_size"],
          num_warmup_steps=transformer_config["num_warmup_steps"])

      optimizer = optimization.get_optimizer(transformer_config, learning_rate)

      global_step = tf.compat.v1.train.get_global_step()

      if not transformer_config["use_bias"]:
        logging.info("Fixing position embedding, i.e. not trainable.")
        posemb = "pegasus/embeddings/position_embeddings"
        tvars = list(filter(lambda v: v.name.split(":")[0] != posemb, tvars))

      gradients = optimizer.compute_gradients(total_loss, tvars)
      train_op = optimizer.apply_gradients(gradients, global_step=global_step)

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          host_call=utils.add_scalars_to_summary(
              transformer_config["output_dir"],
              {"learning_rate": learning_rate}))

    elif mode == tf.estimator.ModeKeys.EVAL:

      tokenizer = tft.SentencepieceTokenizer(
          model=tf.io.gfile.GFile(transformer_config["vocab_model_file"],
                                  "rb").read())

      def rouge_py_func(label_sent, pred_sent):
        """Approximate ROUGE scores, always run externally for final scores."""
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeLsum"],
            use_stemmer=True)
        r1, r2, rl = 0.0, 0.0, 0.0
        for ls, ps in zip(label_sent, pred_sent):
          score = scorer.score(ls.decode("utf-8"), ps.decode("utf-8"))
          r1 += score["rouge1"].fmeasure
          r2 += score["rouge2"].fmeasure
          rl += score["rougeLsum"].fmeasure
        return r1/len(label_sent), r2/len(label_sent), rl/len(label_sent)

      def metric_fn(loss, log_probs, label_ids, pred_ids):
        loss = tf.compat.v1.metrics.mean(values=loss)
        log_probs = tf.compat.v1.metrics.mean(
            values=log_probs,
            weights=tf.cast(tf.not_equal(label_ids, 0), tf.float32))
        metric_dict = {
            "prediction_loss": loss,
            "log_likelihood": log_probs,
        }

        if not transformer_config["use_tpu"]:
          # Approximate ROUGE scores if not running on tpus.
          # Always run externally for final scores.
          label_sent = tokenizer.detokenize(label_ids)
          label_sent = tf.strings.regex_replace(label_sent, r"([<\[]\S+[>\]])",
                                                b" \\1")
          pred_sent = tokenizer.detokenize(pred_ids)
          pred_sent = tf.strings.regex_replace(pred_sent, r"([<\[]\S+[>\]])",
                                               b" \\1")
          if transformer_config["substitute_newline"]:
            label_sent = tf.strings.regex_replace(
                label_sent, transformer_config["substitute_newline"], "\n")
            pred_sent = tf.strings.regex_replace(
                pred_sent, transformer_config["substitute_newline"], "\n")
          rouge_value = tf.compat.v1.py_func(
              func=rouge_py_func,
              inp=[label_sent, pred_sent],
              Tout=[tf.float64, tf.float64, tf.float64],
              stateful=False)
          rouge_value = tf.cast(rouge_value, tf.float32)
          rouge1 = tf.compat.v1.metrics.mean(values=rouge_value[0])
          rouge2 = tf.compat.v1.metrics.mean(values=rouge_value[1])
          rougeL = tf.compat.v1.metrics.mean(values=rouge_value[2])  # pylint: disable=invalid-name

          metric_dict.update({
              "eval/Rouge-1": rouge1,
              "eval/Rouge-2": rouge2,
              "eval/Rouge-L": rougeL,
          })
        return metric_dict

      eval_metrics = (metric_fn,
                      [total_loss, llh, labels, pred_ids])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics)
    else:

      prediction_dict = {"pred_ids": pred_ids}
      if not transformer_config["use_tpu"]:
        tokenizer = tft.SentencepieceTokenizer(
            model=tf.io.gfile.GFile(transformer_config["vocab_model_file"],
                                    "rb").read())
        pred_sent = tokenizer.detokenize(pred_ids)
        # Add a space before special tokens.
        pred_sent = tf.strings.regex_replace(
            pred_sent, r"([<\[]\S+[>\]])", b" \\1")
        if transformer_config["substitute_newline"]:
          pred_sent = tf.strings.regex_replace(
              pred_sent, transformer_config["substitute_newline"], "\n")
        prediction_dict.update({"pred_sent": pred_sent})

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=prediction_dict)

    return output_spec

  return model_fn


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
  """Calculate cross entropy loss while ignoring padding.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
  Returns:
    Returns the cross entropy loss and weight tensors: float32 tensors with
      shape [batch_size, max(length_logits, length_labels)]
  """
  with tf.name_scope("loss"):

    if labels is not None:
      # Calculate smoothing cross entropy
      with tf.name_scope("smoothing_cross_entropy"):
        confidence = 1.0 - smoothing
        vocab_float = tf.cast(vocab_size - 1, tf.float32)
        low_confidence = (1.0 - confidence) / vocab_float
        soft_targets = tf.one_hot(
            labels,
            depth=vocab_size,
            on_value=confidence,
            off_value=low_confidence)
        xentropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=soft_targets)

        # Calculate the best (lowest) possible value of cross entropy, and
        # subtract from the cross entropy loss.
        normalizing_constant = -(
            confidence * tf.math.log(confidence) + vocab_float *
            low_confidence * tf.math.log(low_confidence + 1e-20))
        xentropy -= normalizing_constant

      weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
      loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    else:
      loss = tf.constant(0.0)

    return loss


def main(_):

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_export:
    raise ValueError(
        "At least one of `do_train`, `do_eval` must be True.")

  transformer_config = flags.as_dictionary()

  if FLAGS.max_encoder_length > transformer_config["max_position_embeddings"]:
    raise ValueError(
        "Cannot use sequence length %d because the model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_encoder_length,
         transformer_config["max_position_embeddings"]))

  tf.io.gfile.makedirs(FLAGS.output_dir)
  if FLAGS.do_train:
    flags.save(os.path.join(FLAGS.output_dir, "summarization.config"))

  model_fn = model_fn_builder(transformer_config)
  estimator = utils.get_estimator(transformer_config, model_fn)
  tmp_data_dir = os.path.join(FLAGS.output_dir, "tfds")

  if FLAGS.do_train:
    logging.info("***** Running training *****")
    logging.info("  Batch size = %d", estimator.train_batch_size)
    logging.info("  Num steps = %d", FLAGS.num_train_steps)
    train_input_fn = input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length,
        max_decoder_length=FLAGS.max_decoder_length,
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
        max_decoder_length=FLAGS.max_decoder_length,
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
