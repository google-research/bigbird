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

"""Library for rematerialization.

Incubates a version of tf.recompute_grad that is XLA compatible.
"""
import collections
import numbers
import os
import threading
from typing import Deque, List, NamedTuple, Optional, Sequence, Text, Union

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import custom_gradient


# Remove when https://github.com/tensorflow/tensorflow/pull/45298
# gets merged
def get_variable_by_name(var_name):
  """Retrieves tf.Variable from name in MirroredStrategy (multi-gpu)."""

  # Get all variables, but it will have copies from different replicas
  all_global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

  def _replica_filter(var):
    """Filter out variables from different context."""
    try:
      return var_name == var.op.name
    except AttributeError:
      return False
  candidate_vars = list(filter(_replica_filter, all_global_vars))

  if len(candidate_vars) >= 1:
    # Filter out non-trainable variables.
    candidate_vars = [v for v in candidate_vars if v.trainable]
  else:
    raise ValueError('Unsuccessful at finding variable {}.'.format(var_name))

  if len(candidate_vars) == 1:
    return candidate_vars[0]
  elif len(candidate_vars) > 1:
    raise ValueError(
        'Unsuccessful at finding trainable variable {}. '
        'Number of candidates: {}. '
        'Candidates: {}'.format(var_name, len(candidate_vars), candidate_vars))
  else:
    # The variable is not trainable.
    return None
custom_gradient.get_variable_by_name = get_variable_by_name


class RecomputeContext(
    NamedTuple('RecomputeContext', [
        ('is_recomputing', bool),
        ('seed', tf.Tensor),
        ('children', Deque['RecomputeContext']),
    ])):
  """Context for recomputation.

  Attributes:
    is_recomputing: Whether we are in a recomputation phase.
    seed: Scalar integer tensor that should be used with stateless random ops
      for deterministic behavior and correct computation of the gradient.
    children: Nested `RecomputeContext` instances. Used internally by
      `recompute_grad` to track nested instances of `RecomputeContext`.
  """

  def __enter__(self):
    return _context_stack.push(self)

  def __exit__(self, exc_type, exc_value, traceback):
    _context_stack.pop(self)


# Simplified version of `_DefaultStack` in
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py.
class _ContextStack(threading.local):
  """A thread-local stack for providing implicit recompute contexts."""

  def __init__(self):
    super(_ContextStack, self).__init__()
    self._stack = []

  def top(self) -> Optional[RecomputeContext]:
    return self._stack[-1] if self._stack else None

  def push(self, context: RecomputeContext):
    self._stack.append(context)
    return context

  def pop(self, context: RecomputeContext):
    if self._stack[-1] is not context:
      raise AssertionError('Nesting violated for RecomputeContext.')
    self._stack.pop()


_context_stack = _ContextStack()


def get_recompute_context() -> Optional[RecomputeContext]:
  """Returns the current recomputing context if it exists."""
  return _context_stack.top()


# Adapted from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/control_flow_util.py.
def _get_containing_xla_context(graph: tf.Graph) -> Optional[object]:
  """Returns the first ancestor `XLAControlFlowContext` in the `graph`."""
  ctxt = graph._get_control_flow_context()  # pylint: disable=protected-access
  while ctxt:
    if ctxt.IsXLAContext():
      return ctxt
    ctxt = ctxt.outer_context
  return None


def _in_xla_context(graph: Optional[tf.Graph] = None) -> bool:
  """Detects whether we are in an XLA context."""
  if '--tf_xla_auto_jit=2' in os.environ.get('TF_XLA_FLAGS', ''):
    return True
  graph = tf.compat.v1.get_default_graph() if graph is None else graph
  while True:
    if _get_containing_xla_context(graph) is not None:
      return True
    try:
      graph = graph.outer_graph
    except AttributeError:
      return False


def _force_data_dependency(
    first_compute: Sequence[tf.Tensor],
    then_compute: Sequence[tf.Tensor]) -> List[tf.Tensor]:
  """Force all of `then_compute` to depend on all of `first_compute`.

  Uses a dummy data dependency, which is useful when running on TPUs because
  XLA ignores control dependencies. Only supports float arguments.

  Args:
    first_compute: Sequence of `Tensor`s to be executed before `then_compute`.
    then_compute: Sequence of `Tensor`s to executed after `first_compute`.

  Returns:
    Sequence of `Tensor`s with same length of `then_compute`.

  Raises:
    ValueError: if ranks are unknown or types are not floating.
  """

  def _first_element(x):
    if x.shape.ndims is None:
      raise ValueError('Rank of Tensor %s must be known' % x)
    ndims = x.shape.ndims
    begin = tf.zeros(ndims, dtype=tf.int32)
    size = tf.ones(ndims, dtype=tf.int32)
    return tf.reshape(tf.slice(x, begin, size), [])

  first_compute_sum = tf.add_n(
      [_first_element(x) for x in first_compute if x is not None])
  dtype = first_compute_sum.dtype
  if not dtype.is_floating:
    raise ValueError('_force_data_dependency only supports floating dtypes.')
  zero = np.finfo(dtype.as_numpy_dtype).tiny * first_compute_sum
  return [
      x + tf.cast(zero, x.dtype) if x is not None else None
      for x in then_compute
  ]


def _make_seed_if_none(seed: Optional[tf.Tensor]) -> tf.Tensor:
  """Uses the global generator to make a seed if necessary."""
  if seed is not None:
    return seed
  generator = tf.random.experimental.get_global_generator()
  # The two seeds for stateless random ops don't have individual semantics and
  # are scrambled together, so providing one seed is fine. This makes it easier
  # for users to provide a local seed without worrying about integer overflow.
  # See `make_seeds` in
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/stateful_random_ops.py.
  try:
    return generator.uniform_full_int([], tf.int32, name='recompute_grad_seed')
  except (RuntimeError, TypeError, ValueError, tf.errors.NotFoundError) as e:
    # For a number of reasons, the above operation can fail like using multiple
    # graphs or toggling between eager and graph modes. Reset the generator.
    logging.warn('Resetting the generator. %s: %s', type(e), e)
    tf.random.experimental.set_global_generator(None)
    generator = tf.random.experimental.get_global_generator()
    return generator.uniform_full_int([], tf.int32, name='recompute_grad_seed')


def recompute_grad(f, seed=None):
  """An eager-compatible version of recompute_grad.

  For f(*args, **kwargs), this supports gradients with respect to args, or to
  gradients with respect to any variables residing in the kwarg 'variables'.
  Note that for keras layer and model objects, this is handled automatically.

  Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
  be able to access the member variables of that object, because `g` returns
  through the wrapper function `inner`.  When recomputing gradients through
  objects that inherit from keras, we suggest keeping a reference to the
  underlying object around for the purpose of accessing these variables.

  Args:
    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.
    seed: Optional seed for random ops. `seed` should an integer scalar
      `Tensor`. When compiling to XLA, `seed` must have dtype `tf.int32`. If
      `seed` is not provided one will be generated.

  Returns:
   A function `g` that wraps `f`, but which recomputes `f` on the backwards
   pass of a gradient call.
  """

  @tf.custom_gradient
  def inner(*args, **kwargs):
    """Inner function closure for calculating gradients."""
    # Detect when we're nested and in the backwards pass, so we don't generate
    # an additional seed.
    parent_context = get_recompute_context()
    if parent_context is not None and parent_context.is_recomputing:
      # Use the cached context in the recomputation phase.
      with parent_context.children.popleft()._replace(
          is_recomputing=True) as context:
        result = f(*args, **kwargs)
    else:
      with RecomputeContext(
          is_recomputing=False,
          seed=_make_seed_if_none(seed),
          children=collections.deque()) as context:
        result = f(*args, **kwargs)
        # In the forward pass, build up a tree of recomputation contexts.
        if parent_context is not None and not parent_context.is_recomputing:
          parent_context.children.append(context)

    def grad(*dresult, **grad_kwargs):
      """Gradient function calculation for inner function."""
      variables = grad_kwargs.pop('variables', None)
      if grad_kwargs:
        raise ValueError('Found unexpected kwargs for `grad`: ',
                         list(grad_kwargs.keys()))
      inputs, seed = list(args), context.seed
      if _in_xla_context():
        inputs = _force_data_dependency(
            tf.nest.flatten(dresult), inputs + [seed])
        seed = inputs.pop()
      # tf.keras.backend.set_learning_phase(1)
      with tf.GradientTape() as tape:
        tape.watch(inputs)
        if variables is not None:
          tape.watch(variables)
        with tf.control_dependencies(dresult):
          with context._replace(is_recomputing=True, seed=seed):
            result = f(*inputs, **kwargs)
      kw_vars = []
      if variables is not None:
        kw_vars = list(variables)
      grads = tape.gradient(
          result, list(inputs) + kw_vars, output_gradients=dresult)
      return grads[:len(inputs)], grads[len(inputs):]

    return result, grad

  return inner


######################## STATELESS DROPOUT LAYERS ##############################


def _as_shape(shape: Union[Sequence[int], tf.TensorShape]) -> tf.TensorShape:
  """Converts the given object to a TensorShape."""
  return shape if isinstance(shape, tf.TensorShape) else tf.TensorShape(shape)


def _get_noise_shape(
    x: tf.Tensor, noise_shape: Union[Sequence[int], tf.TensorShape]
) -> Union[tf.Tensor, tf.TensorShape, Sequence[int]]:
  """Computes the shape of the binary mask for dropout."""
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return tf.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = _as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tf.TensorShape(new_dims)

  return noise_shape


def stateless_dropout(x: tf.Tensor,
                      rate: float,
                      seed: tf.Tensor,
                      noise_shape: Optional[Union[Sequence[int],
                                                  tf.TensorShape]] = None,
                      name: Optional[Text] = None) -> tf.Tensor:
  """Computes dropout: randomly sets elements to zero to prevent overfitting.

  See https://www.tensorflow.org/api_docs/python/tf/nn/dropout.
  This version differs in that the seed is required if the rate is nonzero.

  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability that each
      element is dropped. For example, setting rate=0.1 would drop 10% of input
      elements.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
      Must have dtype `tf.int32` when compiling to XLA.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the shape for
      randomly generated keep/drop flags.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` of the same shape of `x`.

  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because the output would be all zeros,
      which is likely not what was intended.
  """
  with tf.name_scope(name or 'stateless_dropout') as name:
    x = tf.convert_to_tensor(x, name='x')
    if not x.dtype.is_floating:
      raise ValueError('x has to be a floating point tensor since it\'s going '
                       ' to be scaled. Got a %s tensor instead.' % x.dtype)
    if isinstance(rate, numbers.Real):
      if not (rate >= 0 and rate < 1):
        raise ValueError('rate must be a scalar tensor or a float in the '
                         'range [0, 1), got %g' % rate)
      if rate > 0.5:
        logging.log_first_n(
            logging.WARN, 'Large dropout rate: %g (>0.5). In TensorFlow '
            '.x, dropout() uses dropout rate instead of keep_prob. '
            'Please ensure that this is intended.', 5, rate)

    # Early return if nothing needs to be dropped.
    if tf.get_static_value(rate) == 0:
      return x

    rate = tf.convert_to_tensor(rate, dtype=x.dtype, name='rate')
    rate.shape.assert_has_rank(0)
    noise_shape = _get_noise_shape(x, noise_shape)
    # Sample a uniform distribution on [0.0, 1.0) and select values larger than
    # rate.
    #
    # NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
    # and subtract 1.0.
    random_tensor = tf.random.stateless_uniform(
        noise_shape, seed=seed, dtype=x.dtype)
    keep_prob = 1 - rate
    scale = 1 / keep_prob
    # NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
    # float to be selected, hence we use a >= comparison.
    keep_mask = random_tensor >= rate
    ret = x * scale * tf.cast(keep_mask, x.dtype)
    if not tf.executing_eagerly():
      ret.set_shape(x.get_shape())
    return ret


# Reimplements internal function
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/smart_cond.py.
def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Arguments:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if not callable(true_fn):
    raise TypeError('`true_fn` must be callable.')
  if not callable(false_fn):
    raise TypeError('`false_fn` must be callable.')
  pred_value = tf.get_static_value(pred)
  if isinstance(pred, tf.Variable) or pred_value is None:
    return tf.cond(
        pred, true_fn=true_fn, false_fn=false_fn, name=name)
  if pred_value:
    return true_fn()
  else:
    return false_fn()


# See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout.
class RecomputingDropout(tf.keras.layers.Layer):
  """`tf.keras.layers.Dropout` that supports `recompute_grad`."""

  def __init__(self,
               rate,
               noise_shape=None,
               seed=None,
               force_recomputation=False,
               **kwargs):
    """Initializes `RecomputingDropout`.

    Args:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D integer tensor representing the shape of the binary
        dropout mask that will be multiplied with the input. For instance, if
        inputs have shape `(batch_size, timesteps, features)` and you want the
        dropout mask to be the same for all timesteps, you can use
        `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.
      force_recomputation: If `True`, then raises an error if called outside a
        recompute context.
      **kwargs: Keyword arguments for `tf.keras.layers.Layer`.
    """

    super(RecomputingDropout, self).__init__(**kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.force_recomputation = force_recomputation
    self.supports_masking = True
    # Create a layer-specific seed to combine with the global recompute seed.
    self._recompute_seed = (
        np.random.randint(-2**31, 2**31, dtype=np.int32)
        if seed is None else seed)

  def _get_noise_shape(self, inputs):
    # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    # which will override `self.noise_shape`, and allows for custom noise
    # shapes with dynamically sized inputs.
    if self.noise_shape is None:
      return None

    concrete_inputs_shape = tf.shape(inputs)
    noise_shape = []
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return tf.convert_to_tensor(noise_shape)

  def call(self, inputs, training=None):
    """Builds computation graph.

    Args:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      `inputs` masked according to layer configuration.

    Raises:
      ValueError: If `force_recomputation` is `True` and called outside a
        a recompute context.
    """
    if self.rate == 0:
      return inputs

    if training is None:
      training = tf.keras.backend.learning_phase()

    def dropped_inputs():
      """Randomly drops elements of `inputs` when `training=True`."""
      recompute_context = get_recompute_context()
      if recompute_context is None:
        if self.force_recomputation:
          raise ValueError(
              'RecomputeContext is required when force_recomputation=True.')
        return tf.nn.dropout(
            inputs,
            noise_shape=self._get_noise_shape(inputs),
            seed=self.seed,
            rate=self.rate)
      seed = tf.stack([recompute_context.seed, self._recompute_seed])
      return stateless_dropout(
          inputs,
          rate=self.rate,
          seed=seed,
          noise_shape=self._get_noise_shape(inputs))

    output = smart_cond(training, dropped_inputs, lambda: tf.identity(inputs))
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed,
        'force_recomputation': self.force_recomputation,
    }
    base_config = super(RecomputingDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
