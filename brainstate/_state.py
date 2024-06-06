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

import contextlib
import threading
from typing import Any, Tuple, Dict, List, Callable

import jax
import numpy as np
from jax.api_util import shaped_abstractify
from jax.extend import source_info_util

from .util import DictManager

PyTree = Any
max_int = np.iinfo(np.int32)

__all__ = [
  'State', 'ShortTermState', 'LongTermState', 'ParamState',
  'StateDictManager', 'visible_state_dict',
  'check_state_value_tree',
]

_pytree_registered_objects = set()


def _register_pytree_cls(cls):
  if cls not in _pytree_registered_objects:
    jax.tree_util.register_pytree_node_class(cls)
    _pytree_registered_objects.add(cls)


# The global state of the state stack is accessed by a thread-local object.
# This allows concurrent tracing in separate threads; passing traced objects
# between threads is forbidden.
class ThreadLocalStack(threading.local):
  def __init__(self):
    self.stack: List[StateTrace] = []


thread_local_stack = ThreadLocalStack()

_global_context_to_check_state_tree = [False]


@contextlib.contextmanager
def check_state_value_tree() -> None:
  """
  The contex manager to check weather the tree structure of the state value keeps consistently.
  """
  try:
    _global_context_to_check_state_tree.append(True)
    yield
  finally:
    _global_context_to_check_state_tree.pop()


class State(object):
  """
  The pointer to specify the dynamical data.

  To implement a new subclass of :py:class:`~.State`, you only need to inherent this class:

  Example::

    class MyState(State):
      pass

  The typical examples of :py:class:`~.State` subclass are:

  - :py:class:`~.ShortTermState`: The short-term state, which is used to store the short-term data in the program.
  - :py:class:`~.LongTermState`: The long-term state, which is used to store the long-term data in the program.
  - :py:class:`~.ParamState`: The parameter state, which is used to store the parameters in the program.
  - :py:class:`~.RandomState`: The random generator state, which is used to store the random key in the program.

  Args:
    value: PyTree. It can be anything as a pyTree.
  """
  __module__ = 'brainstate'
  __slots__ = ('_value', '_tree', '_level', '_source_info', '_check_tree')

  def __init__(self, value: PyTree):
    if isinstance(value, State):
      value = value.value
    self._value = value
    self._tree = jax.tree.structure(value)
    self._check_tree = False
    self._level = len(thread_local_stack.stack)
    self._source_info = source_info_util.current()

  @property
  def value(self) -> PyTree:
    """
    The data and its value.
    """
    self._check_if_deleted()

    # read the value by the stack (>= level)
    trace: StateTrace
    for trace in thread_local_stack.stack[self._level:]:
      trace.read_its_value(self)
    # return the value
    return self._value

  @value.setter
  def value(self, v) -> None:
    """
    Set the value of the state.

    Args:
      v: The value.
    """
    # value checking
    v = v.value if isinstance(v, State) else v
    self._check_value(v)
    # write the value by the stack (>= level)
    trace: StateTrace
    for trace in thread_local_stack.stack[self._level:]:
      trace.write_its_value(self)
    # set the value
    self._value = v

  def _check_value(self, v):
    if self._check_tree or _global_context_to_check_state_tree[-1]:
      in_tree = jax.tree_util.tree_structure(v)
      if in_tree != self._tree:
        self._raise_error_with_source_info(
          ValueError(f'The given value {in_tree} does not '
                     f'match with the origin tree structure '
                     f'{self._tree}.')
        )

  def _raise_error_with_source_info(self, error: Exception):
    name_stack = source_info_util.current_name_stack() + self.source_info.name_stack
    with source_info_util.user_context(self.source_info.traceback, name_stack=name_stack):
      raise error

  def _check_if_deleted(self):
    pass

  @property
  def source_info(self) -> source_info_util.SourceInfo:
    """
    The source information of the state, can be useful to identify
    the source code where the definition of the state.

    Returns:
      The source information.
    """
    return self._source_info

  def tree_flatten(self):
    """Flattens this variable.

    Returns:
      A pair where the first element is a list of leaf values
      and the second element is a treedef representing the
      structure of the flattened tree.
    """
    return (self._value,), (self._level,)

  @classmethod
  def tree_unflatten(cls, aux_data, flat_contents):
    """Reconstructs a variable from the aux_data and the leaves.

    Args:
      aux_data:
      flat_contents:

    Returns:
      The variable.
    """
    (_level,) = aux_data
    self = cls(flat_contents[0])
    self._level = max_int
    return self

  def __repr__(self):
    leaves, tree = jax.tree.flatten(self._value)
    leaves_info = [ShapeDtype(leaf.shape, leaf.dtype) for leaf in leaves]
    tree_info = jax.tree.unflatten(tree, leaves_info)
    return f'{self.__class__.__name__}({tree_info})'


class ShapeDtype:
  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def __repr__(self):
    return f'{self.dtype}{list(self.shape)}'


class ShortTermState(State):
  """
  The short-term state, which is used to store the short-term data in the program.

  For example, in a training process, the gradients of the model are short-term states.
  """

  __module__ = 'brainstate'


class LongTermState(State):
  """
  The long-term state, which is used to store the long-term data in the program.

  For example, in a training process, the weights of the model are long-term states.

  """

  __module__ = 'brainstate'


class ParamState(LongTermState):
  """
  The parameter state, which is used to store the trainable parameters in the model.
  """
  __module__ = 'brainstate'


class StateDictManager(DictManager):
  """
  State stack, for collecting all :py:class:`~.State` used in the program.

  :py:class:`~.StateDictManager` supports all features of python dict.
  """

  __module__ = 'brainstate'

  def assign_values(self, *args: Dict) -> None:
    """
    Assign the value for each element according to the given ``data``.
    """
    for arg in args:
      assert isinstance(arg, dict), 'Must be an instance of dict.'
      for k, v in arg.items():
        self._set_elem(k, v)

  def split_values(self, *filters: type) -> Tuple[Dict, ...]:
    """
    Split the values into several subsets of stack by the given types.
    """
    results = tuple(DictManager() for _ in range(len(filters) + 1))
    for k, v in self.items():
      for i, filt in enumerate(filters):
        if isinstance(v, filt):
          results[i][k] = v.value
          break
      else:
        results[-1][k] = v.value
    return results

  def collect_values(self) -> Dict:
    """
    Collect the values by the given types.
    """
    results = DictManager()
    for k, v in self.items():
      results[k] = v.value
    return results

  def split(self, first: type, *others: type) -> Tuple['StateDictManager', ...]:
    return super().split(first, *others)

  def to_dict_values(self) -> Dict:
    """
    Convert the values into a dict.
    """
    return {k: v.value for k, v in self.items()}

  def _check_elem(self, elem):
    assert isinstance(elem, State), f'must be instance of {State}'

  def _set_elem(self, key: Any, value: Any) -> None:
    self[key].value = value


class visible_state_dict(StateDictManager):
  """
  The state dictionary whose elements are visible to ``.states()`` collection functions.
  """
  pass


class StateTrace(object):
  """
  The state trace, which is used to trace the states automatically.
  """

  def __init__(self, new_arg: Callable = None):
    self.states: List[State] = []
    self.types: List[str] = []
    self._id2index = dict()
    self._org_values = []
    self._jax_trace_new_arg = new_arg
    self._written_ids = set()

  def set_new_arg(self, new_arg: Callable) -> None:
    self._jax_trace_new_arg = new_arg

  def new_arg(self, state: State) -> None:
    if self._jax_trace_new_arg is not None:
      # internal use
      state._value = jax.tree.map(lambda x: self._jax_trace_new_arg(shaped_abstractify(x)), state._value)

  def __enter__(self) -> 'StateTrace':
    thread_local_stack.stack.append(self)
    return self

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    thread_local_stack.stack.pop()

  def read_its_value(self, state: State) -> None:
    """
    Read the value of the state.

    Args:
      state: The state.
    """
    id_ = id(state)
    if id_ not in self._id2index:
      self._id2index[id_] = len(self.states)
      self.states.append(state)
      self.types.append('read')
      self._org_values.append(state._value)  # internal use
      self.new_arg(state)

  def write_its_value(self, state: State) -> None:
    """
    Write the value of the state.

    Args:
      state: The state.
    """
    id_ = id(state)
    if id_ not in self._id2index:
      self.read_its_value(state)
    if id_ not in self._written_ids:
      index = self._id2index[id_]
      self.types[index] = 'write'
      self._written_ids.add(id_)

  def collect_values(self, *categories: str) -> Tuple:
    """
    Collect the values by the given categories.

    Args:
      *categories: The categories.

    Returns:
      results: The values.
    """
    results = []
    for st, ty in zip(self.states, self.types):
      if ty in categories:
        results.append(st.value)
    return tuple(results)

  def recovery_original_values(self) -> None:
    """
    Recovery the original values.
    """
    for st, val in zip(self.states, self._org_values):
      # internal use
      st._value = val
