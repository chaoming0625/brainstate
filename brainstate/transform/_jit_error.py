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

import functools
from functools import partial
from typing import Callable, Union

import jax
from jax import numpy as jnp
from jax.core import Primitive, ShapedArray
from jax.interpreters import batching, mlir

from brainstate._utils import set_module_as

__all__ = [
  'jit_error',
]


@set_module_as('brainstate.transform')
def remove_vmap(x, op: str = 'any'):
  if op == 'any':
    return _any_without_vmap(x)
  elif op == 'all':
    return _all_without_vmap(x)
  elif op == 'none':
    return _without_vmap(x)
  else:
    raise ValueError(f'Do not support type: {op}')


def _without_vmap(x):
  return _no_vmap_prim.bind(x)


def _without_vmap_imp(x):
  return x


def _without_vmap_abs(x):
  return x


def _without_vmap_batch(x, batch_axes):
  (x,) = x
  return _without_vmap(x), batching.not_mapped


_no_vmap_prim = Primitive('no_vmap')
_no_vmap_prim.def_impl(_without_vmap_imp)
_no_vmap_prim.def_abstract_eval(_without_vmap_abs)
batching.primitive_batchers[_no_vmap_prim] = _without_vmap_batch
mlir.register_lowering(_no_vmap_prim, mlir.lower_fun(_without_vmap_imp, multiple_results=False))


def _any_without_vmap(x):
  return _any_no_vmap_prim.bind(x)


def _any_without_vmap_imp(x):
  return jnp.any(x)


def _any_without_vmap_abs(x):
  return ShapedArray(shape=(), dtype=jnp.bool_)


def _any_without_vmap_batch(x, batch_axes):
  (x,) = x
  return _any_without_vmap(x), batching.not_mapped


_any_no_vmap_prim = Primitive('any_no_vmap')
_any_no_vmap_prim.def_impl(_any_without_vmap_imp)
_any_no_vmap_prim.def_abstract_eval(_any_without_vmap_abs)
batching.primitive_batchers[_any_no_vmap_prim] = _any_without_vmap_batch
mlir.register_lowering(_any_no_vmap_prim, mlir.lower_fun(_any_without_vmap_imp, multiple_results=False))


def _all_without_vmap(x):
  return _all_no_vmap_prim.bind(x)


def _all_without_vmap_imp(x):
  return jnp.all(x)


def _all_without_vmap_abs(x):
  return ShapedArray(shape=(), dtype=jnp.bool_)


def _all_without_vmap_batch(x, batch_axes):
  (x,) = x
  return _all_without_vmap(x), batching.not_mapped


_all_no_vmap_prim = Primitive('all_no_vmap')
_all_no_vmap_prim.def_impl(_all_without_vmap_imp)
_all_no_vmap_prim.def_abstract_eval(_all_without_vmap_abs)
batching.primitive_batchers[_all_no_vmap_prim] = _all_without_vmap_batch
mlir.register_lowering(_all_no_vmap_prim, mlir.lower_fun(_all_without_vmap_imp, multiple_results=False))


def _err_jit_true_branch(err_fun, args, kwargs):
  jax.debug.callback(err_fun, *args, **kwargs)


def _err_jit_false_branch(args, kwargs):
  pass


def _error_msg(msg, *arg, **kwargs):
  if len(arg):
    msg = msg % arg
  if len(kwargs):
    msg = msg.format(**kwargs)
  raise ValueError(msg)


@set_module_as('brainstate.transform')
def jit_error(
    pred,
    err_fun: Union[Callable, str],
    *err_args,
    **err_kwargs,
):
  """
  Check errors in a jit function.

  Examples
  --------

  It can give a function which receive arguments that passed from the JIT variables and raise errors.

  >>> def error(x):
  >>>    raise ValueError(f'error {x}')
  >>> x = jax.random.uniform(jax.random.PRNGKey(0), (10,))
  >>> jit_error(x.sum() < 5., error, x)

  Or, it can be a simple string message.

  >>> x = jax.random.uniform(jax.random.PRNGKey(0), (10,))
  >>> jit_error(x.sum() < 5., "Error: the sum is less than 5. Got {s}", s=x.sum())


  Parameters
  ----------
  pred: bool, Array
    The boolean prediction.
  err_fun: callable
    The error function, which raise errors.
  err_args: 
    The arguments which passed into `err_f`.
  err_kwargs: 
    The keywords which passed into `err_f`.
  """
  if isinstance(err_fun, str):
    err_fun = partial(_error_msg, err_fun)

  jax.lax.cond(
    remove_vmap(pred, op='any'),
    partial(_err_jit_true_branch, err_fun),
    _err_jit_false_branch,
    jax.tree.map(functools.partial(remove_vmap, op='none'), err_args),
    jax.tree.map(functools.partial(remove_vmap, op='none'), err_kwargs),
  )
