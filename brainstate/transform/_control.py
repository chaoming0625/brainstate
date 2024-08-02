# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
# ==============================================================================

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import wraps, reduce
from typing import Callable, TypeVar, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from brainstate._utils import set_module_as
from ._jit_error import jit_error, remove_vmap
from ._make_jaxpr import StatefulFunction, _assign_state_values
from ._progress_bar import ProgressBar

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')
T = TypeVar('T')
BooleanNumeric = Any  # A bool, or a Boolean array.

__all__ = [
  'cond', 'switch', 'ifelse', 'scan', 'for_loop', 'while_loop',
]


def _wrapped_fun(stateful_fun: StatefulFunction, states, return_states=True):
  @wraps(stateful_fun.fun)
  def wrapped_branch(state_vals, *operands):
    assert len(states) == len(state_vals)
    for st, val in zip(states, state_vals):
      st.value = val
    out = stateful_fun.jaxpr_call_auto(*operands)
    if return_states:
      return tuple(st.value for st in states), out
    return out

  return wrapped_branch


@set_module_as('brainstate.transform')
def cond(pred, true_fun: Callable, false_fun: Callable, *operands):
  """
  Conditionally apply ``true_fun`` or ``false_fun``.

  Provided arguments are correctly typed, ``cond()`` has equivalent
  semantics to this Python implementation, where ``pred`` must be a
  scalar type::

    def cond(pred, true_fun, false_fun, *operands):
      if pred:
        return true_fun(*operands)
      else:
        return false_fun(*operands)


  In contrast with :func:`jax.lax.select`, using ``cond`` indicates that only one of
  the two branches is executed (up to compiler rewrites and optimizations).
  However, when transformed with :func:`~jax.vmap` to operate over a batch of
  predicates, ``cond`` is converted to :func:`~jax.lax.select`.

  Args:
    pred: Boolean scalar type, indicating which branch function to apply.
    true_fun: Function (A -> B), to be applied if ``pred`` is True.
    false_fun: Function (A -> B), to be applied if ``pred`` is False.
    operands: Operands (A) input to either branch depending on ``pred``. The
      type can be a scalar, array, or any pytree (nested Python tuple/list/dict)
      thereof.

  Returns:
    Value (B) of either ``true_fun(*operands)`` or ``false_fun(*operands)``,
    depending on the value of ``pred``. The type can be a scalar, array, or any
    pytree (nested Python tuple/list/dict) thereof.
  """
  if not (callable(true_fun) and callable(false_fun)):
    raise TypeError("true_fun and false_fun arguments should be callable.")

  if pred is None:
    raise TypeError("cond predicate is None")
  if isinstance(pred, Sequence) or np.ndim(pred) != 0:
    raise TypeError(f"Pred must be a scalar, got {pred} of " +
                    (f"type {type(pred)}" if isinstance(pred, Sequence)
                     else f"shape {np.shape(pred)}."))

  # check pred
  try:
    pred_dtype = jax.dtypes.result_type(pred)
  except TypeError as err:
    raise TypeError("Pred type must be either boolean or number, got {}.".format(pred)) from err
  if pred_dtype.kind != 'b':
    if pred_dtype.kind in 'iuf':
      pred = pred != 0
    else:
      raise TypeError("Pred type must be either boolean or number, got {}.".format(pred_dtype))

  # not jit
  if jax.config.jax_disable_jit and isinstance(jax.core.get_aval(pred), jax.core.ConcreteArray):
    if pred:
      return true_fun(*operands)
    else:
      return false_fun(*operands)

  # evaluate jaxpr
  true_fun_wrap = StatefulFunction(true_fun).make_jaxpr(*operands)
  false_fun_wrap = StatefulFunction(false_fun).make_jaxpr(*operands)

  # wrap the functions
  all_states = tuple(set(true_fun_wrap.get_states() + false_fun_wrap.get_states()))
  true_fun = _wrapped_fun(true_fun_wrap, all_states)
  false_fun = _wrapped_fun(false_fun_wrap, all_states)

  # operands
  operands = ([st.value for st in all_states],) + operands

  # cond
  state_vals, out = jax.lax.cond(pred, true_fun, false_fun, *operands)
  _assign_state_values(all_states, state_vals)
  return out

  # ops, ops_tree = jax.tree.flatten(operands)
  # linear_ops = [False] * len(ops)
  # ops_avals = tuple(jax.util.safe_map(_abstractify, ops))
  #
  # # true and false jaxprs
  # jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
  #   (true_fun, false_fun), ops_tree, ops_avals, 'cond')
  # if any(isinstance(op_aval, state.AbstractRef) for op_aval in ops_avals):
  #   raise ValueError("Cannot pass `Ref`s into `cond`.")
  # true_jaxpr, false_jaxpr = jaxprs
  # out_tree, false_out_tree = out_trees
  # if any(isinstance(out_aval, state.AbstractRef) for out_aval in true_jaxpr.out_avals + false_jaxpr.out_avals):
  #   raise ValueError("Cannot return `Ref`s from `cond`.")
  #
  # _check_tree_and_avals("true_fun and false_fun output",
  #                       out_tree, true_jaxpr.out_avals,
  #                       false_out_tree, false_jaxpr.out_avals)
  # joined_effects = jax.core.join_effects(true_jaxpr.effects, false_jaxpr.effects)
  # disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(joined_effects)
  # if disallowed_effects:
  #   raise NotImplementedError(f'Effects not supported in `cond`: {disallowed_effects}')
  #
  # # replace jaxpr effects
  # index = jax.lax.convert_element_type(pred, np.int32)
  # if joined_effects:
  #   # Raise index in case of effects to allow data-dependence-based discharging
  #   # of those effects (even if they don't have an explicit data dependence).
  #   index = jax.core.raise_as_much_as_possible(index)
  # false_jaxpr = _replace_jaxpr_effects(false_jaxpr, joined_effects)
  # true_jaxpr = _replace_jaxpr_effects(true_jaxpr, joined_effects)
  #
  # # bind
  # linear = [False] * len(consts) + linear_ops
  # cond_outs = jax.lax.cond_p.bind(index, *consts, *ops, branches=(false_jaxpr, true_jaxpr), linear=tuple(linear))
  #
  # # outputs
  # st_vals, out = jax.tree.unflatten(out_tree, cond_outs)
  # for st, val in zip(all_states, st_vals):
  #   st.value = val
  # return out


@set_module_as('brainstate.transform')
def switch(index, branches: Sequence[Callable], *operands):
  """
  Apply exactly one of ``branches`` given by ``index``.

  If ``index`` is out of bounds, it is clamped to within bounds.

  Has the semantics of the following Python::

    def switch(index, branches, *operands):
      index = clamp(0, index, len(branches) - 1)
      return branches[index](*operands)

  Internally this wraps XLA's `Conditional
  <https://www.tensorflow.org/xla/operation_semantics#conditional>`_
  operator. However, when transformed with :func:`~jax.vmap` to operate over a
  batch of predicates, ``cond`` is converted to :func:`~jax.lax.select`.

  Args:
    index: Integer scalar type, indicating which branch function to apply.
    branches: Sequence of functions (A -> B) to be applied based on ``index``.
    operands: Operands (A) input to whichever branch is applied.

  Returns:
    Value (B) of ``branch(*operands)`` for the branch that was selected based
    on ``index``.
  """
  # check branches
  if not all(callable(branch) for branch in branches):
    raise TypeError("branches argument should be a sequence of callables.")

  # check index
  if len(np.shape(index)) != 0:
    raise TypeError(f"Branch index must be scalar, got {index} of shape {np.shape(index)}.")
  try:
    index_dtype = jax.dtypes.result_type(index)
  except TypeError as err:
    msg = f"Index type must be an integer, got {index}."
    raise TypeError(msg) from err
  if index_dtype.kind not in 'iu':
    raise TypeError(f"Index type must be an integer, got {index} as {index_dtype}")

  # format branches
  branches = tuple(branches)
  if len(branches) == 0:
    raise ValueError("Empty branch sequence")
  elif len(branches) == 1:
    return branches[0](*operands)

  # format index
  index = jax.lax.convert_element_type(index, np.int32)
  lo = np.array(0, np.int32)
  hi = np.array(len(branches) - 1, np.int32)
  index = jax.lax.clamp(lo, index, hi)

  # not jit
  if jax.config.jax_disable_jit and isinstance(jax.core.core.get_aval(index), jax.core.ConcreteArray):
    return branches[int(index)](*operands)

  # evaluate jaxpr
  wrapped_branches = [StatefulFunction(branch) for branch in branches]
  for wrapped_branch in wrapped_branches:
    wrapped_branch.make_jaxpr(*operands)

  # wrap the functions
  all_states = tuple(set(reduce(operator.add, [wrapped_branch.get_states() for wrapped_branch in wrapped_branches])))
  branches = tuple(_wrapped_fun(wrapped_branch, all_states) for wrapped_branch in wrapped_branches)

  # operands
  operands = ([st.value for st in all_states],) + operands

  # switch
  state_vals, out = jax.lax.switch(index, branches, *operands)
  _assign_state_values(all_states, state_vals)
  return out

  # ops, ops_tree = jax.tree.flatten(operands)
  # ops_avals = tuple(jax.util.safe_map(_abstractify, ops))
  #
  # # true jaxprs
  # jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
  #   branches, ops_tree, ops_avals, primitive_name='switch')
  # for i, (out_tree, jaxpr) in enumerate(zip(out_trees[1:], jaxprs[1:])):
  #   _check_tree_and_avals(f"branch 0 and {i + 1} outputs",
  #                         out_trees[0], jaxprs[0].out_avals,
  #                         out_tree, jaxpr.out_avals)
  # joined_effects = jax.core.join_effects(*(jaxpr.effects for jaxpr in jaxprs))
  # disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(joined_effects)
  # if disallowed_effects:
  #   raise NotImplementedError(f'Effects not supported in `switch`: {disallowed_effects}')
  # if joined_effects:
  #   # Raise index in case of effects to allow data-dependence-based discharging
  #   # of those effects (even if they don't have an explicit data dependence).
  #   index = jax.core.raise_as_much_as_possible(index)
  #
  # # bind
  # linear = (False,) * (len(consts) + len(ops))
  # cond_outs = jax.lax.cond_p.bind(index, *consts, *ops, branches=tuple(jaxprs), linear=linear)
  #
  # # outputs
  # st_vals, out = jax.tree.unflatten(out_trees[0], cond_outs)
  # for st, val in zip(all_states, st_vals):
  #   st.value = val
  # return out


@set_module_as('brainstate.transform')
def ifelse(conditions, branches, *operands, check_cond: bool = True):
  """
  ``If-else`` control flows looks like native Pythonic programming.

  Examples
  --------

  >>> import brainstate as bst
  >>> def f(a):
  >>>    return bst.transform.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
  >>>                               branches=[lambda: 1,
  >>>                                         lambda: 2,
  >>>                                         lambda: 3,
  >>>                                         lambda: 4,
  >>>                                         lambda: 5])
  >>> f(1)
  4
  >>> f(0)
  5

  Parameters
  ----------
  conditions: bool, sequence of bool, Array
    The boolean conditions.
  branches: Any
    The branches, at least has two elements. Elements can be functions,
    arrays, or numbers. The number of ``branches`` and ``conditions`` has
    the relationship of `len(branches) == len(conditions) + 1`.
    Each branch should receive one arguement for ``operands``.
  *operands: optional, Any
    The operands for each branch.
  check_cond: bool
    Whether to check the conditions. Default is True.

  Returns
  -------
  res: Any
    The results of the control flow.
  """
  # check branches
  if not all(callable(branch) for branch in branches):
    raise TypeError("branches argument should be a sequence of callables.")

  # format branches
  branches = tuple(branches)
  if len(branches) == 0:
    raise ValueError("Empty branch sequence")
  elif len(branches) == 1:
    return branches[0](*operands)
  if len(conditions) != len(branches):
    raise ValueError("The number of conditions should be equal to the number of branches.")

  # format index
  conditions = jnp.asarray(conditions, np.int32)
  if check_cond:
    jit_error(jnp.sum(conditions) != 1, "Only one condition can be True. But got {}.", err_arg=conditions)
  index = jnp.where(conditions, size=1, fill_value=len(conditions) - 1)[0][0]
  return switch(index, branches, *operands)


def _wrap_fun_with_pbar(fun, pbar_runner):
  @wraps(fun)
  def new_fun(new_carry, inputs):
    i, old_carry = new_carry
    old_carry, old_outputs = fun(old_carry, inputs)
    pbar_runner(remove_vmap(i, op='none'))
    return (i + 1, old_carry), old_outputs

  return new_fun


def _wrapped_scan_fun(stateful_fun: StatefulFunction, states):
  @wraps(stateful_fun.fun)
  def wrapped_fun(new_carry, inputs):
    state_vals, carry = new_carry
    assert len(states) == len(state_vals)
    for st, val in zip(states, state_vals):
      st.value = val
    carry, out = stateful_fun.jaxpr_call_auto(carry, inputs)
    return (tuple(st.value for st in states), carry), out

  return wrapped_fun


@set_module_as('brainstate.transform')
def scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    pbar: ProgressBar | None = None,
) -> tuple[Carry, Y]:
  """
  Scan a function over leading array axes while carrying along state.

  The `Haskell-like type signature`_ in brief is

  .. code-block:: haskell

    scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

  where for any array type specifier ``t``, ``[t]`` represents the type with an additional
  leading axis, and if ``t`` is a pytree (container) type with array leaves then ``[t]``
  represents the type with the same pytree structure and corresponding leaves
  each with an additional leading axis.

  When the type of ``xs`` (denoted `a` above) is an array type or None, and the type
  of ``ys`` (denoted `b` above) is an array type, the semantics of :func:`~scan` are
  given roughly by this Python implementation::

    def scan(f, init, xs, length=None):
      if xs is None:
        xs = [None] * length
      carry = init
      ys = []
      for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
      return carry, np.stack(ys)

  Unlike that Python version, both ``xs`` and ``ys`` may be arbitrary pytree
  values, and so multiple arrays can be scanned over at once and produce multiple
  output arrays. ``None`` is actually a special case of this, as it represents an
  empty pytree.

  Also unlike that Python version, :func:`~scan` is a JAX primitive and is
  lowered to a single WhileOp. That makes it useful for reducing
  compilation times for JIT-compiled functions, since native Python
  loop constructs in an :func:`~jax.jit` function are unrolled, leading to large
  XLA computations.

  Finally, the loop-carried value ``carry`` must hold a fixed shape and dtype
  across all iterations (and not just be consistent up to NumPy rank/shape
  broadcasting and dtype promotion rules, for example). In other words, the type
  ``c`` in the type signature above represents an array with a fixed shape and
  dtype (or a nested tuple/list/dict container data structure with a fixed
  structure and arrays with fixed shape and dtype at the leaves).

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    init: an initial loop carry value of type ``c``, which can be a scalar,
      array, or any pytree (nested Python tuple/list/dict) thereof, representing
      the initial loop carry value. This value must have the same structure as
      the first element of the pair returned by ``f``.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    reverse: optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.
    unroll: optional positive int or bool specifying, in the underlying
      operation of the scan primitive, how many scan iterations to unroll within
      a single iteration of a loop. If an integer is provided, it determines how
      many unrolled loop iterations to run within a single rolled iteration of
      the loop. If a boolean is provided, it will determine if the loop is
      completely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e.
      `unroll=False`).
    pbar: optional :class:`~.ProgressBar` instance to display the progress
      of the scan operation.

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.

  .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
  """
  # check "f"
  if not callable(f):
    raise TypeError("f argument should be a callable.")

  # check "xs"
  xs_flat, xs_tree = jax.tree.flatten(xs)
  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError as err:
    raise ValueError("scan got value with no leading axis to scan over: "
                     "{}.".format(', '.join(str(x) for x in xs_flat if not hasattr(x, 'shape')))) from err
  if length is not None:
    length = int(length)
    if not all(length == l for l in lengths):
      raise ValueError(("scan got `length` argument of {} which disagrees with "
                        "leading axis sizes {}.").format(length, [x.shape[0] for x in xs_flat]))
  else:
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
      msg = "scan got values with different leading axis sizes: {}."
      raise ValueError(msg.format(', '.join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
      raise ValueError("scan got no values to scan over and `length` not provided.")
    else:
      length, = unique_lengths

  # function with progress bar
  has_pbar = False
  if pbar is not None:
    has_pbar = True
    f = _wrap_fun_with_pbar(f, pbar.init(length))
    init = (0, init) if pbar else init

  # not jit
  if jax.config.jax_disable_jit:
    if length == 0:
      raise ValueError("zero-length scan is not supported in disable_jit() mode because the output type is unknown.")
    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(length)):
      xs_slice = [jax.lax.index_in_dim(x, i, keepdims=False) for x in xs_flat]
      carry, y = f(carry, jax.tree.unflatten(xs_tree, xs_slice))
      ys.append(y)
    stacked_y = jax.tree.map(lambda *elems: jnp.stack(elems), *maybe_reversed(ys))
    if has_pbar:
      return carry[1], stacked_y
    else:
      return carry, stacked_y

  # evaluate jaxpr, get all states #
  # ------------------------------ #
  xs_avals = [jax.core.raise_to_shaped(jax.core.get_aval(x)) for x in xs_flat]
  x_avals = [jax.core.mapped_aval(length, 0, aval) for aval in xs_avals]
  stateful_fun = StatefulFunction(f).make_jaxpr(init, xs_tree.unflatten(x_avals))
  all_states = stateful_fun.get_states()
  wrapped_f = _wrapped_scan_fun(stateful_fun, all_states)

  # scan
  init = (tuple(st.value for st in all_states), init)
  (state_vals, carry), ys = jax.lax.scan(wrapped_f, init, xs, length=length, reverse=reverse, unroll=unroll)
  _assign_state_values(all_states, state_vals)
  if has_pbar:
    carry = carry[1]
  return carry, ys


def _forloop_to_scan_fun(f: Callable):
  @wraps(f)
  def scan_fun(carry, x):
    return carry, f(*x)

  return scan_fun


@set_module_as('brainstate.transform')
def for_loop(
    f,
    *xs,
    length: Optional[int] = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    pbar: Optional[ProgressBar] = None
):
  """
  ``for-loop`` control flow with :py:class:`~.State`.

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    reverse: optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.
    unroll: optional positive int or bool specifying, in the underlying
      operation of the scan primitive, how many scan iterations to unroll within
      a single iteration of a loop. If an integer is provided, it determines how
      many unrolled loop iterations to run within a single rolled iteration of
      the loop. If a boolean is provided, it will determine if the loop is
      completely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e.
      `unroll=False`).
    pbar: optional :class:`~.ProgressBar` instance to display the progress
      of the scan operation.

  Returns:
    The return represents the stacked outputs of the second output of ``f`` 
    when scanned over the leading axis of the inputs.

  """
  _, ys = scan(_forloop_to_scan_fun(f),
               init=None,
               xs=xs,
               length=length,
               reverse=reverse,
               unroll=unroll,
               pbar=pbar)
  return ys


@set_module_as('brainstate.transform')
def while_loop(
    cond_fun: Callable[[T], BooleanNumeric],
    body_fun: Callable[[T], T],
    init_val: T
) -> T:
  """
  Call ``body_fun`` repeatedly in a loop while ``cond_fun`` is True.

  The `Haskell-like type signature`_ in brief is

  .. code-block:: haskell

    while_loop :: (a -> Bool) -> (a -> a) -> a -> a

  The semantics of ``while_loop`` are given by this Python implementation::

    def while_loop(cond_fun, body_fun, init_val):
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val

  Unlike that Python version, ``while_loop`` is a JAX primitive and is lowered
  to a single WhileOp. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Also unlike the Python analogue, the loop-carried value ``val`` must hold a
  fixed shape and dtype across all iterations (and not just be consistent up to
  NumPy rank/shape broadcasting and dtype promotion rules, for example). In
  other words, the type ``a`` in the type signature above represents an array
  with a fixed shape and dtype (or a nested tuple/list/dict container data
  structure with a fixed structure and arrays with fixed shape and dtype at the
  leaves).

  Another difference from using Python-native loop constructs is that
  ``while_loop`` is not reverse-mode differentiable because XLA computations
  require static bounds on memory requirements.

  Args:
    cond_fun: function of type ``a -> Bool``.
    body_fun: function of type ``a -> a``.
    init_val: value of type ``a``, a type that can be a scalar, array, or any
      pytree (nested Python tuple/list/dict) thereof, representing the initial
      loop carry value.

  Returns:
    The output from the final iteration of body_fun, of type ``a``.

  .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
  """
  if not (callable(body_fun) and callable(cond_fun)):
    raise TypeError("while_loop: body_fun and cond_fun arguments should be callable.")
  if jax.config.jax_disable_jit:
    try:
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val
    except jax.core.ConcretizationTypeError:
      # Can't run this while_loop in Python (e.g. because there's a vmap
      # transformation on it), so we fall back to the primitive version.
      pass

  # evaluate jaxpr
  stateful_cond = StatefulFunction(cond_fun).make_jaxpr(init_val)
  stateful_body = StatefulFunction(body_fun).make_jaxpr(init_val)
  all_states = tuple(set(stateful_cond.get_states() + stateful_body.get_states()))
  new_cond_fun = _wrapped_fun(stateful_cond, all_states, return_states=False)
  new_body_fun = _wrapped_fun(stateful_body, all_states, return_states=True)

  # while_loop
  state_vals, final_val = jax.lax.while_loop(new_cond_fun,
                                             new_body_fun,
                                             (tuple(st.value for st in all_states), init_val))
  _assign_state_values(all_states, state_vals)
  return final_val
