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

"""Common flag definitions."""

import json
import sys

from absl import flags
from absl import logging
import bigbird
import tensorflow.compat.v2 as tf

import sentencepiece as spm

# pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 9):
  import importlib.resources as importlib_resources
else:
  import importlib_resources


############################### FLAGS UTILS ####################################

FLAGS = flags.FLAGS
DEFINE_bool = flags.DEFINE_bool
DEFINE_enum = flags.DEFINE_enum
DEFINE_float = flags.DEFINE_float
DEFINE_integer = flags.DEFINE_integer
DEFINE_string = flags.DEFINE_string


# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.

# Basic model config flags

flags.DEFINE_float(
    "attention_probs_dropout_prob", 0.1,
    "The dropout probability for attention coefficients when using original.")
flags.DEFINE_string(
    "hidden_act", "gelu",
    "The non-linear activation function (function or string) in the encoder "
    "and pooler.")
flags.DEFINE_float(
    "hidden_dropout_prob", 0.1,
    "The dropout probability for all fully connected layers in the embeddings, "
    "encoder, decoder, and pooler.")
flags.DEFINE_integer(
    "hidden_size", 768,
    "Size of the transformer layers and the pooler layer.")
flags.DEFINE_float(
    "initializer_range", 0.02,
    "The stdev of the truncated_normal_initializer for initializing all "
    "weight matrices.")
flags.DEFINE_integer(
    "intermediate_size", 3072,
    "The size of intermediate (i.e. feed-forward) layer in the Transformer.")
flags.DEFINE_integer(
    "max_position_embeddings", 4096,
    "The size position embeddings of matrix, which dictates the maximum"
    "length for which the model can be run.")
flags.DEFINE_integer(
    "num_attention_heads", 12,
    "Number of attention heads for each attention layer in the Transformer.")
flags.DEFINE_integer(
    "num_hidden_layers", 12,
    "Number of hidden layers in the model (same for encoder and decoder).")
flags.DEFINE_integer(
    "type_vocab_size", 2,
    "The vocabulary size of the `token_type_ids`.")
flags.DEFINE_bool(
    "use_bias", True,
    "Whether to use bias for key/query/value.")
flags.DEFINE_bool(
    "rescale_embedding", False,
    "Whether to rescale word embedding by hidden dimensions.")
flags.DEFINE_bool(
    "use_gradient_checkpointing", False,
    "Whether to recompute encoder fwd pass during back prop for saving memory.")
flags.DEFINE_string(
    "scope", "bert",
    "Variable scope name.")
flags.DEFINE_string(
    "vocab_model_file", "gpt2",
    "The sentence piece model for vocabulary. Shortcuts for standard "
    "gpt2 and pegasus vocabs are their name respectively.")

# Simulated and Block attention settings

flags.DEFINE_enum(
    "attention_type", "block_sparse",
    ["original_full", "simulated_sparse", "block_sparse"],
    "Selecting attention implementation. "
    "'original_full': full attention from original bert. "
    "'simulated_sparse': simulated sparse attention. "
    "'block_sparse': blocked implementation of sparse attention.")
flags.DEFINE_enum(
    "norm_type", "postnorm",
    ["prenorm", "postnorm"],
    "Selecting when to apply layer-norm. "
    "'prenorm': Before attention layer, e.g. Pegasus. "
    "'postnorm': After attention layer, e.g. Bert.")
flags.DEFINE_integer(
    "block_size", 16,
    "The block size for the attention mask.")
flags.DEFINE_integer(
    "num_rand_blocks", 3,
    "Number of random blocks per row.")

# Adaptive optimizer configs

flags.DEFINE_float(
    "weight_decay_rate", 0.01,
    "L2 penalty as weight decay to be used.")

flags.DEFINE_float(
    "optimizer_beta1", 0.9,
    "The exponential decay rate for the 1st moment estimates.")

flags.DEFINE_float(
    "optimizer_beta2", 0.999,
    "The exponential decay rate for the 2nd moment estimates.")

flags.DEFINE_float(
    "optimizer_epsilon", 1e-6,
    "Adaptivty trade-off parameter.")

# TPU settings

flags.DEFINE_bool(
    "use_tpu", False,
    "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "tpu_job_name", None,
    "Name of TPU worker, if anything other than 'tpu_worker'")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "master", None,
    "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string(
    "iterations_per_loop", "1000",
    "How many steps to make in each estimator call.")


def as_dictionary():
  """Get current config from flag."""

  # Resolve vocab file location from hotword
  if FLAGS.vocab_model_file == "gpt2":
    FLAGS.vocab_model_file = str(importlib_resources.files(bigbird).joinpath(
        "vocab/gpt2.model"))
  elif FLAGS.vocab_model_file == "pegasus":
    FLAGS.vocab_model_file = str(importlib_resources.files(bigbird).joinpath(
        "vocab/pegasus.model"))

  config = {
      # transformer basic configs
      "attention_probs_dropout_prob": FLAGS.attention_probs_dropout_prob,
      "hidden_act": FLAGS.hidden_act,
      "hidden_dropout_prob": FLAGS.hidden_dropout_prob,
      "hidden_size": FLAGS.hidden_size,
      "initializer_range": FLAGS.initializer_range,
      "intermediate_size": FLAGS.intermediate_size,
      "max_position_embeddings": FLAGS.max_position_embeddings,
      "num_attention_heads": FLAGS.num_attention_heads,
      "num_hidden_layers": FLAGS.num_hidden_layers,
      "type_vocab_size": FLAGS.type_vocab_size,
      "scope": FLAGS.scope,
      "use_bias": FLAGS.use_bias,
      "rescale_embedding": FLAGS.rescale_embedding,
      "use_gradient_checkpointing": FLAGS.use_gradient_checkpointing,
      "vocab_model_file": FLAGS.vocab_model_file,
      # sparse mask configs
      "attention_type": FLAGS.attention_type,
      "norm_type": FLAGS.norm_type,
      "block_size": FLAGS.block_size,
      "num_rand_blocks": FLAGS.num_rand_blocks,
      # common bert configs
      "data_dir": FLAGS.data_dir,
      "output_dir": FLAGS.output_dir,
      "init_checkpoint": FLAGS.init_checkpoint,
      "max_encoder_length": FLAGS.max_encoder_length,
      "substitute_newline": FLAGS.substitute_newline,
      "do_train": FLAGS.do_train,
      "do_eval": FLAGS.do_eval,
      "do_export": FLAGS.do_export,
      "train_batch_size": FLAGS.train_batch_size,
      "eval_batch_size": FLAGS.eval_batch_size,
      "optimizer": FLAGS.optimizer,
      "learning_rate": FLAGS.learning_rate,
      "num_train_steps": FLAGS.num_train_steps,
      "num_warmup_steps": FLAGS.num_warmup_steps,
      "save_checkpoints_steps": FLAGS.save_checkpoints_steps,
      "weight_decay_rate": FLAGS.weight_decay_rate,
      "optimizer_beta1": FLAGS.optimizer_beta1,
      "optimizer_beta2": FLAGS.optimizer_beta2,
      "optimizer_epsilon": FLAGS.optimizer_epsilon,
      # TPU settings
      "use_tpu": FLAGS.use_tpu,
      "tpu_name": FLAGS.tpu_name,
      "tpu_zone": FLAGS.tpu_zone,
      "tpu_job_name": FLAGS.tpu_job_name,
      "gcp_project": FLAGS.gcp_project,
      "master": FLAGS.master,
      "num_tpu_cores": FLAGS.num_tpu_cores,
      "iterations_per_loop": FLAGS.iterations_per_loop,
  }

  # pretraining dedicated flags
  if hasattr(FLAGS, "max_predictions_per_seq"):
    config["max_predictions_per_seq"] = FLAGS.max_predictions_per_seq
  if hasattr(FLAGS, "masked_lm_prob"):
    config["masked_lm_prob"] = FLAGS.masked_lm_prob
  if hasattr(FLAGS, "max_eval_steps"):
    config["max_eval_steps"] = FLAGS.max_eval_steps
  if hasattr(FLAGS, "preprocessed_data"):
    config["preprocessed_data"] = FLAGS.preprocessed_data
  if hasattr(FLAGS, "use_nsp"):
    config["use_nsp"] = FLAGS.use_nsp

  # classifier dedicated flags
  if hasattr(FLAGS, "num_labels"):
    config["num_labels"] = FLAGS.num_labels

  # summarization dedicated flags
  if hasattr(FLAGS, "max_decoder_length"):
    config["max_decoder_length"] = FLAGS.max_decoder_length
  if hasattr(FLAGS, "trainable_bias"):
    config["trainable_bias"] = FLAGS.trainable_bias
  if hasattr(FLAGS, "couple_encoder_decoder"):
    config["couple_encoder_decoder"] = FLAGS.couple_encoder_decoder
  if hasattr(FLAGS, "beam_size"):
    config["beam_size"] = FLAGS.beam_size
  if hasattr(FLAGS, "alpha"):
    config["alpha"] = FLAGS.alpha
  if hasattr(FLAGS, "label_smoothing"):
    config["label_smoothing"] = FLAGS.label_smoothing

  # calculate vocab
  sp_model = spm.SentencePieceProcessor()
  sp_proto = tf.io.gfile.GFile(config["vocab_model_file"], "rb").read()
  sp_model.LoadFromSerializedProto(sp_proto)
  vocab_size = sp_model.GetPieceSize()
  config["vocab_size"] = vocab_size

  return config


def save(path):
  """Save current flag config."""
  config = as_dictionary()
  with tf.io.gfile.GFile(path, "w") as f:
    json.dump(config, f, indent=4, sort_keys=True)

  # log flags
  max_len = max([len(ii) for ii in config.keys()])
  fmt_string = "\t%" + str(max_len) + "s : %s"
  logging.info("Arguments:")
  for key, value in sorted(config.items()):
    logging.info(fmt_string, key, value)

  return config


def load(path):
  """Set flag from saved config."""

  with tf.io.gfile.GFile(path) as f:
    config = json.load(f)

  # log and set flags
  max_len = max([len(ii) for ii in config.keys()])
  fmt_string = "\t%" + str(max_len) + "s : %s"
  logging.info("Arguments:")
  for key, value in config.items():
    if hasattr(FLAGS, key):
      logging.info(fmt_string, key, value)
      setattr(FLAGS, key, value)

  return config
