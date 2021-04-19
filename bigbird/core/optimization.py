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

"""Functions and classes related to optimization (weight updates)."""

import re

from absl import logging
import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import resource_variable_ops


def get_linear_warmup_linear_decay_lr(init_lr, num_train_steps,
                                      num_warmup_steps):
  """Calculate learning rate with linear warmup and linear decay."""
  global_step = tf.compat.v1.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.compat.v1.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_step, tf.float32)
    warmup_steps_float = tf.cast(num_warmup_steps, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  return learning_rate


def get_linear_warmup_rsqrt_decay_lr(init_lr, hidden_size,
                                     num_warmup_steps):
  """Calculate learning rate with linear warmup and rsqrt decay."""
  num_warmup_steps = tf.cast(num_warmup_steps, tf.float32)
  global_step = tf.compat.v1.train.get_or_create_global_step()
  global_step = tf.cast(global_step, tf.float32)

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
  learning_rate *= tf.math.rsqrt(tf.cast(hidden_size, tf.float32))
  # Apply linear warmup
  learning_rate *= tf.minimum(1.0, global_step / num_warmup_steps)
  # Apply rsqrt decay
  learning_rate *= tf.math.rsqrt(tf.maximum(global_step, num_warmup_steps))

  return learning_rate


def get_optimizer(params, learning_rate):
  """Gets the optimzer based on the hparams and current mode (TPU vs. CPU/GPU).

  Args:
      params: A dictionary containing training hyperparameters.
      learning_rate: A float32 scalar.

  Returns:
    A string or an optimizer instance.
  """
  optimizer = None

  if params["optimizer"] == "Adafactor":
    try:
      from tensor2tensor.utils import adafactor  # pylint: disable=g-import-not-at-top
      optimizer = adafactor.AdafactorOptimizer(learning_rate=learning_rate)
    except ImportError:
      logging.error("tensor2tensor not installed. Cannot use Adafactor."
                    "Defaulting to Adam.")
      params["optimizer"] = "Adam"

  if params["optimizer"] == "Adam":
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate,
        beta1=params["optimizer_beta1"],
        beta2=params["optimizer_beta2"],
        epsilon=params["optimizer_epsilon"])

  if params["optimizer"] == "AdamWeightDecay":
    optimizer = AdamWeightDecayOptimizer(
        learning_rate,
        weight_decay_rate=params["weight_decay_rate"],
        beta_1=params["optimizer_beta1"],
        beta_2=params["optimizer_beta2"],
        epsilon=params["optimizer_epsilon"],
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if params["optimizer"] == "SGD":
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)

  if optimizer is None:
    raise ValueError("Unknown optimizer: {}.".format(params["optimizer"]))

  if params["use_tpu"]:
    # Average the gradients across TPU cores.
    optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

  return optimizer


class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _apply_dense(self, grad, var):
    param_name = self._get_variable_name(var.name)
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")

    # Standard Adam update.
    next_m = (
        tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
    next_v = (
        tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                  tf.square(grad)))

    update = next_m / (tf.sqrt(next_v) + self.epsilon)

    # Just adding the square of the weights to the loss function is *not*
    # the correct way of using L2 regularization/weight decay with Adam,
    # since that will interact with the m and v parameters in strange ways.
    #
    # Instead we want ot decay the weights in a manner that doesn't interact
    # with the m/v parameters. This is equivalent to adding the square
    # of the weights to the loss with plain (non-momentum) SGD.
    if self._do_use_weight_decay(param_name):
      update += self.weight_decay_rate * var

    update_with_lr = self.learning_rate * update

    next_param = var - update_with_lr

    return tf.group(
        [var.assign(next_param),
         m.assign(next_m),
         v.assign(next_v)])

  def _resource_apply_dense(self, grad, var):
    """See `tf.train.Optimizer._resource_apply_dense()`."""
    return self._apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    """See `tf.train.Optimizer._apply_sparse()`."""
    def scatter_update_fn(x, i, v):
      return tf.compat.v1.scatter_update(x, i, v, use_locking=self._use_locking)
    return self._apply_sparse_shared(
        grad.values, grad.indices, var, scatter_update_fn)

  def _resource_apply_sparse(self, grad, var, indices):
    """See `tf.train.Optimizer._resource_apply_spase()`."""
    def scatter_update_fn(x, i, v):
      with tf.control_dependencies(
          [resource_variable_ops.resource_scatter_update(x.handle, i, v)]):
        return x.value()
    return self._apply_sparse_shared(grad, indices, var, scatter_update_fn)

  def _apply_sparse_shared(self, grad, indices, var, scatter_update_fn):
    """Applies sparse gradients to a variable.

    Args:
      grad: A tensor for the `values` of `tf.IndexedSlices`.
      indices: A tensor for the `indices` of `tf.IndexedSlices`.
      var: A `tf.Variable` object.
      scatter_update_fn: A function which performs scattered update to
        a `tf.Variable` object. It takes tuple of (x, i, v) where:
          * x: A `tf.Variable` object which is updated by `i` and `v`,
          * i: A tensor for the `indices` of `tf.IndexedSlices`,
          * v: A tensor for the `values` of `tf.IndexedSlices`,
        and returns a tensor after updating `x`.

    Returns:
      An op which updates `var` with `grad` and `indices`.
    """
    param_name = self._get_variable_name(var.name)
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")

    # m_t = beta1 * m + (1 - beta1) * g_t
    m_scaled_g_values = tf.multiply(1.0 - self.beta_1, grad)
    m_t = m.assign(m * self.beta_1)
    with tf.control_dependencies([m_t]):
      m_slice = tf.gather(m, indices) + m_scaled_g_values
      m_t = scatter_update_fn(m, indices, m_slice)

    # v_t = beta2 * v + (1 - beta2) * g_t^2
    v_scaled_g_values = tf.multiply(1.0 - self.beta_2, tf.square(grad))
    v_t = v.assign(v * self.beta_2)
    with tf.control_dependencies([v_t]):
      v_slice = tf.gather(v, indices) + v_scaled_g_values
      v_t = scatter_update_fn(v, indices, v_slice)

    update = m_t / (tf.sqrt(v_t) + self.epsilon)

    # Just adding the square of the weights to the loss function is *not*
    # the correct way of using L2 regularization/weight decay with Adam,
    # since that will interact with the m and v parameters in strange ways.
    #
    # Instead we want ot decay the weights in a manner that doesn't interact
    # with the m/v parameters. This is equivalent to adding the square
    # of the weights to the loss with plain (non-momentum) SGD.
    if self._do_use_weight_decay(param_name):
      update += self.weight_decay_rate * var

    update_with_lr = self.learning_rate * update

    next_param = var - update_with_lr

    return tf.group([var.assign(next_param), m_t, v_t])

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
