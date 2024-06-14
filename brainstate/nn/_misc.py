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

from enum import Enum
from functools import wraps
from typing import Sequence, Callable

import brainunit as bu
import jax.numpy as jnp

from .. import environ
from .._state import State
from ..transform import vector_grad

__all__ = [
  # 'exp_euler',
  'exp_euler_step',
]

git_issue_addr = 'https://github.com/brainpy/brainscale/issues'


def state_traceback(states: Sequence[State]):
  """
  Traceback the states of the brain model.

  Parameters
  ----------
  states : Sequence[bst.State]
    The states of the brain model.

  Returns
  -------
  str
    The traceback information of the states.
  """
  state_info = []
  for i, state in enumerate(states):
    state_info.append(f'State {i}: {state}\n'
                      f'defined at \n'
                      f'{state.source_info.traceback}\n')
  return '\n'.join(state_info)


class BaseEnum(Enum):
  @classmethod
  def get_by_name(cls, name: str):
    for item in cls:
      if item.name == name:
        return item
    raise ValueError(f'Cannot find the {cls.__name__} type {name}.')

  @classmethod
  def get(cls, type_: str | Enum):
    if isinstance(type_, cls):
      return type_
    elif isinstance(type_, str):
      return cls.get_by_name(type_)
    else:
      raise ValueError(f'Cannot find the {cls.__name__} type {type_}.')


def exp_euler(fun):
  """
  Exponential Euler method for solving ODEs.

  Args:
    fun: Callable. The function to be solved.

  Returns:
    The integral function.
  """

  @wraps(fun)
  def integral(*args, **kwargs):
    assert len(args) > 0, 'The input arguments should not be empty.'
    if args[0].dtype not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
      raise ValueError(
        'The input data type should be float32, float64, float16, or bfloat16 '
        'when using Exponential Euler method.'
        f'But we got {args[0].dtype}.'
      )
    dt = environ.get('dt')
    linear, derivative = vector_grad(fun, argnums=0, return_value=True)(*args, **kwargs)
    phi = bu.math.exprel(dt * linear)
    return args[0] + dt * phi * derivative

  return integral


def exp_euler_step(fun: Callable, *args, **kwargs):
  """
  Exponential Euler method for solving ODEs.

  Examples
  --------
  >>> def fun(x, t):
  ...     return -x
  >>> x = 1.0
  >>> exp_euler_step(fun, x, None)

  Args:
    fun: Callable. The function to be solved.

  Returns:
    The integral function.
  """
  assert len(args) > 0, 'The input arguments should not be empty.'
  if args[0].dtype not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
    raise ValueError(
      'The input data type should be float32, float64, float16, or bfloat16 '
      'when using Exponential Euler method.'
      f'But we got {args[0].dtype}.'
    )
  dt = environ.get('dt')
  linear, derivative = vector_grad(fun, argnums=0, return_value=True)(*args, **kwargs)
  phi = bu.math.exprel(dt * linear)
  return args[0] + dt * phi * derivative
