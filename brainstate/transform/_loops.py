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

import math
from collections.abc import Callable
from functools import wraps
from typing import Any, Optional, TypeVar

import jax
import jax.lax as lax
import jax.numpy as jnp

from brainstate._utils import set_module_as
from ._make_jaxpr import StatefulFunction, _assign_state_values
from ._progress_bar import ProgressBar

X = TypeVar('X')
Y = TypeVar('Y')
T = TypeVar('T')
Carry = TypeVar('Carry')
BooleanNumeric = Any  # A bool, or a Boolean array.

__all__ = [
  'scan', 'for_loop', 'while_loop',
  'bounded_while_loop',
]


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


def _while_loop_to_scan(cond_fun, body_fun, val, max_steps, base):
  if max_steps == 1:
    return body_fun(val)
  else:

    def call(val_):
      return _while_loop_to_scan(cond_fun, body_fun, val_, max_steps // base, base)

    def scan_fn(val_, _):
      return lax.cond(cond_fun(val_), call, lambda x: x, val_), None

    # Don't put checkpointing on the lowest level
    if max_steps != base:
      scan_fn = jax.checkpoint(scan_fn, prevent_cse=False)  # pyright: ignore

    return lax.scan(scan_fn, val, xs=None, length=base)[0]


def bounded_while_loop(
    cond_fun: Callable[[T], BooleanNumeric],
    body_fun: Callable[[T], T],
    init_val: T,
    *,
    max_steps: int,
    base: int = 16,
):
  """
  While loop with a bound on the maximum number of steps.

  This function is useful when you want to ensure that a while loop terminates
  even if the condition function is never false. The function is implemented
  using a scan operation, so it is reverse-mode differentiable.

  Args:
    cond_fun: A function of type ``a -> Bool``.
    body_fun: A function of type ``a -> a``.
    init_val: The initial value of type ``a``.
    max_steps: A bound on the maximum number of steps, after which the loop
      terminates unconditionally.
    base: Run time will increase slightly as `base` increases. Compilation time will
      decrease substantially as `math.ceil(math.log(max_steps, base))` decreases.
      (Which happens as `base` increases.)

  Returns:
    The final value, as if computed by a `lax.while_loop`.
  """

  # checking
  if not isinstance(max_steps, int) or max_steps < 0:
    raise ValueError("max_steps must be a non-negative integer")
  init_val = jax.tree.map(jnp.array, init_val)
  if max_steps == 0:
    return init_val

  # evaluate jaxpr
  stateful_cond = StatefulFunction(cond_fun).make_jaxpr(init_val)
  stateful_body = StatefulFunction(body_fun).make_jaxpr(init_val)
  all_states = tuple(set(stateful_cond.get_states() + stateful_body.get_states()))
  new_cond_fun = _wrapped_fun(stateful_cond, all_states, return_states=False)
  new_body_fun = _wrapped_fun(stateful_body, all_states, return_states=True)

  # initial value
  init_val = (tuple(st.value for st in all_states), init_val)

  # while_loop
  rounded_max_steps = base ** int(math.ceil(math.log(max_steps, base)))
  state_vals, val = _while_loop_to_scan(new_cond_fun, new_body_fun, init_val, rounded_max_steps, base)
  _assign_state_values(all_states, state_vals)
  return val
