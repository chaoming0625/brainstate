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

import copy
import functools
import gc
import types
import warnings
from collections.abc import Iterable
from typing import Any, Callable, Tuple, Union, Sequence

import jax
from jax.lib import xla_bridge

from ._utils import set_module_as


__all__ = [
  'unique_name',
  'clear_buffer_memory',
  'not_instance_eval',
  'is_instance_eval',
  'DictManager',
  'MemScaling',
  'IdMemScaling',
  'DotDict',
]

_name2id = dict()
_typed_names = {}


@set_module_as('brainstate.util')
def check_name_uniqueness(name, obj):
  """Check the uniqueness of the name for the object type."""
  if not name.isidentifier():
    raise ValueError(
      f'"{name}" isn\'t a valid identifier '
      f'according to Python language definition. '
      f'Please choose another name.'
    )
  if name in _name2id:
    if _name2id[name] != id(obj):
      raise ValueError(
        f'In BrainPy, each object should have a unique name. '
        f'However, we detect that {obj} has a used name "{name}". \n'
        f'If you try to run multiple trials, you may need \n\n'
        f'>>> brainpy.brainpy_object.clear_name_cache() \n\n'
        f'to clear all cached names. '
      )
  else:
    _name2id[name] = id(obj)


def get_unique_name(type_: str):
  """Get the unique name for the given object type."""
  if type_ not in _typed_names:
    _typed_names[type_] = 0
  name = f'{type_}{_typed_names[type_]}'
  _typed_names[type_] += 1
  return name


@set_module_as('brainstate.util')
def unique_name(name=None, self=None):
  """Get the unique name for this object.

  Parameters
  ----------
  name : str, optional
    The expected name. If None, the default unique name will be returned.
    Otherwise, the provided name will be checked to guarantee its uniqueness.
  self : str, optional
    The name of this class, used for object naming.

  Returns
  -------
  name : str
    The unique name for this object.
  """
  if name is None:
    assert self is not None, 'If name is None, self should be provided.'
    return get_unique_name(type_=self.__class__.__name__)
  else:
    check_name_uniqueness(name=name, obj=self)
    return name


@set_module_as('brainstate.util')
def clear_name_cache(ignore_warn: bool = True):
  """Clear the cached names."""
  _name2id.clear()
  _typed_names.clear()
  if not ignore_warn:
    warnings.warn(f'All named models and their ids are cleared.', UserWarning)


@jax.tree_util.register_pytree_node_class
class DictManager(dict):
  """
  DictManager, for collecting all pytree used in the program.

  :py:class:`~.DictManager` supports all features of python dict.
  """
  __module__ = 'brainstate.util'

  def subset(self, sep: Union[type, Tuple[type, ...], Callable]) -> 'DictManager':
    """
    Get a new stack with the subset of keys.
    """
    gather = type(self)()
    if isinstance(sep, types.FunctionType):
      for k, v in self.items():
        if sep(v):
          gather[k] = v
      return gather
    else:
      for k, v in self.items():
        if isinstance(v, sep):
          gather[k] = v
    return gather

  def not_subset(self, sep: Union[type, Tuple[type, ...]]) -> 'DictManager':
    """
    Get a new stack with the subset of keys.
    """
    gather = type(self)()
    for k, v in self.items():
      if not isinstance(v, sep):
        gather[k] = v
    return gather

  def add_unique_elem(self, key: Any, var: Any):
    """Add a new element."""
    self._check_elem(var)
    if key in self:
      if id(var) != id(self[key]):
        raise ValueError(f'{key} has been registered by {self[key]}, the new value is different from it.')
    else:
      self[key] = var

  def unique(self) -> 'DictManager':
    """
    Get a new type of collections with unique values.

    If one value is assigned to two or more keys,
    then only one pair of (key, value) will be returned.
    """
    gather = type(self)()
    seen = set()
    for k, v in self.items():
      if id(v) not in seen:
        seen.add(id(v))
        gather[k] = v
    return gather

  def assign(self, *args) -> None:
    """
    Assign the value for each element according to the given ``data``.
    """
    for arg in args:
      assert isinstance(arg, dict), 'Must be an instance of dict.'
      for k, v in arg.items():
        self[k] = v

  def split(self, first: type, *others: type) -> Tuple['DictManager', ...]:
    """
    Split the stack into subsets of stack by the given types.
    """
    filters = (first, *others)
    results = tuple(type(self)() for _ in range(len(filters) + 1))
    for k, v in self.items():
      for i, filt in enumerate(filters):
        if isinstance(v, filt):
          results[i][k] = v
          break
      else:
        results[-1][k] = v
    return results

  def pop_by_keys(self, keys: Iterable):
    """
    Pop the elements by the keys.
    """
    for k in tuple(self.keys()):
      if k in keys:
        self.pop(k)

  def pop_by_values(self, values: Iterable, by: str = 'id'):
    """
    Pop the elements by the values.

    Args:
      values: The value ids.
      by: str. The discard method, can be ``id`` or ``value``. Default is 'id'.
    """
    if by == 'id':
      value_ids = {id(v) for v in values}
      for k in tuple(self.keys()):
        if id(self[k]) in value_ids:
          self.pop(k)
    elif by == 'value':
      for k in tuple(self.keys()):
        if self[k] in values:
          self.pop(k)
    else:
      raise ValueError(f'Unsupported method: {by}')

  def difference_by_keys(self, keys: Iterable):
    """
    Get the difference of the stack by the keys.
    """
    return type(self)({k: v for k, v in self.items() if k not in keys})

  def difference_by_values(self, values: Iterable, by: str = 'id'):
    """
    Get the difference of the stack by the values.

    Args:
      values: The value ids.
      by: str. The discard method, can be ``id`` or ``value``. Default is 'id'.
    """
    if by == 'id':
      value_ids = {id(v) for v in values}
      return type(self)({k: v for k, v in self.items() if id(v) not in value_ids})
    elif by == 'value':
      return type(self)({k: v for k, v in self.items() if v not in values})
    else:
      raise ValueError(f'Unsupported method: {by}')

  def intersection_by_keys(self, keys: Iterable):
    """
    Get the intersection of the stack by the keys.
    """
    return type(self)({k: v for k, v in self.items() if k in keys})

  def intersection_by_values(self, values: Iterable, by: str = 'id'):
    """
    Get the intersection of the stack by the values.

    Args:
      values: The value ids.
      by: str. The discard method, can be ``id`` or ``value``. Default is 'id'.
    """
    if by == 'id':
      value_ids = {id(v) for v in values}
      return type(self)({k: v for k, v in self.items() if id(v) in value_ids})
    elif by == 'value':
      return type(self)({k: v for k, v in self.items() if v in values})
    else:
      raise ValueError(f'Unsupported method: {by}')

  def union_by_value_ids(self, other: dict):
    """
    Union the stack by the value ids.

    Args:
      other:

    Returns:

    """

  def __add__(self, other: dict):
    """
    Compose other instance of dict.
    """
    new_dict = type(self)(self)
    new_dict.update(other)
    return new_dict

  def tree_flatten(self):
    return tuple(self.values()), tuple(self.keys())

  @classmethod
  def tree_unflatten(cls, keys, values):
    return cls(jax.util.safe_zip(keys, values))

  def _check_elem(self, elem: Any):
    raise NotImplementedError

  def to_dict(self):
    """
    Convert the stack to a dict.

    Returns
    -------
    dict
      The dict object.
    """
    return dict(self)

  def __copy__(self):
    return type(self)(self)


@set_module_as('brainstate.util')
def clear_buffer_memory(
    platform: str = None,
    array: bool = True,
    compilation: bool = False,
):
  """Clear all on-device buffers.

  This function will be very useful when you call models in a Python loop,
  because it can clear all cached arrays, and clear device memory.

  .. warning::

     This operation may cause errors when you use a deleted buffer.
     Therefore, regenerate data always.

  Parameters
  ----------
  platform: str
    The device to clear its memory.
  array: bool
    Clear all buffer array. Default is True.
  compilation: bool
    Clear compilation cache. Default is False.

  """
  if array:
    for buf in xla_bridge.get_backend(platform).live_buffers():
      buf.delete()
  if compilation:
    jax.clear_caches()
  gc.collect()


class MemScaling(object):
  """
  The scaling object for membrane potential.

  The scaling object is used to transform the membrane potential range to a
  standard range. The scaling object can be used to transform the membrane
  potential to a standard range, and transform the standard range to the
  membrane potential.

  """
  __module__ = 'brainstate.util'

  def __init__(self, scale, bias):
    self._scale = scale
    self._bias = bias

  @classmethod
  def transform(
      cls,
      oring_range: Sequence[Union[float, int]],
      target_range: Sequence[Union[float, int]] = (0., 1.)
  ) -> 'MemScaling':
    """Transform the membrane potential range to a ``Scaling`` instance.

    Args:
      oring_range:   [V_min, V_max]
      target_range:  [scaled_V_min, scaled_V_max]

    Returns:
      The instanced scaling object.
    """
    V_min, V_max = oring_range
    scaled_V_min, scaled_V_max = target_range
    scale = (V_max - V_min) / (scaled_V_max - scaled_V_min)
    bias = scaled_V_min * scale - V_min
    return cls(scale=scale, bias=bias)

  def scale_offset(self, x, bias=None, scale=None):
    """
    Transform the membrane potential to the standard range.

    Parameters
    ----------
    x : array_like
      The membrane potential.
    bias : float, optional
      The bias of the scaling object. If None, the default bias will be used.
    scale : float, optional
      The scale of the scaling object. If None, the default scale will be used.

    Returns
    -------
    x : array_like
      The standard range of the membrane potential.
    """
    if bias is None:
      bias = self._bias
    if scale is None:
      scale = self._scale
    return (x + bias) / scale

  def scale(self, x, scale=None):
    """
    Transform the membrane potential to the standard range.

    Parameters
    ----------
    x : array_like
      The membrane potential.
    scale : float, optional
      The scale of the scaling object. If None, the default scale will be used.

    Returns
    -------
    x : array_like
      The standard range of the membrane potential.
    """
    if scale is None:
      scale = self._scale
    return x / scale

  def offset(self, x, bias=None):
    """
    Transform the membrane potential to the standard range.

    Parameters
    ----------
    x : array_like
      The membrane potential.
    bias : float, optional
      The bias of the scaling object. If None, the default bias will be used.

    Returns
    -------
    x : array_like
      The standard range of the membrane potential.
    """
    if bias is None:
      bias = self._bias
    return x + bias

  def rev_scale(self, x, scale=None):
    """
    Reversely transform the standard range to the original membrane potential.

    Parameters
    ----------
    x : array_like
      The standard range of the membrane potential.
    scale : float, optional
      The scale of the scaling object. If None, the default scale will be used.

    Returns
    -------
    x : array_like
      The original membrane potential.
    """
    if scale is None:
      scale = self._scale
    return x * scale

  def rev_offset(self, x, bias=None):
    """
    Reversely transform the standard range to the original membrane potential.

    Parameters
    ----------
    x : array_like
      The standard range of the membrane potential.
    bias : float, optional
      The bias of the scaling object. If None, the default bias will be used.

    Returns
    -------
    x : array_like
      The original membrane potential.
    """
    if bias is None:
      bias = self._bias
    return x - bias

  def rev_scale_offset(self, x, bias=None, scale=None):
    """
    Reversely transform the standard range to the original membrane potential.

    Parameters
    ----------
    x : array_like
      The standard range of the membrane potential.
    bias : float, optional
      The bias of the scaling object. If None, the default bias will be used.
    scale : float, optional
      The scale of the scaling object. If None, the default scale will be used.

    Returns
    -------
    x : array_like
      The original membrane potential.
    """
    if bias is None:
      bias = self._bias
    if scale is None:
      scale = self._scale
    return x * scale - bias

  def clone(self):
    """
    Clone the scaling object.

    Returns
    -------
    scaling : MemScaling
      The cloned scaling object.
    """
    return MemScaling(bias=self._bias, scale=self._scale)


class IdMemScaling(MemScaling):
  """
  The identity scaling object.

  The identity scaling object is used to transform the membrane potential to
  the standard range, and reversely transform the standard range to the
  membrane potential.

  """
  __module__ = 'brainstate.util'

  def __init__(self):
    super().__init__(scale=1., bias=0.)

  def scale_offset(self, x, bias=None, scale=None):
    """
    Transform the membrane potential to the standard range.
    """
    return x

  def scale(self, x, scale=None):
    """
    Transform the membrane potential to the standard range.
    """
    return x

  def offset(self, x, bias=None):
    """
    Transform the membrane potential to the standard range.
    """
    return x

  def rev_scale(self, x, scale=None):
    """
    Reversely transform the standard range to the original membrane potential.

    """
    return x

  def rev_offset(self, x, bias=None):
    """
    Reversely transform the standard range to the original membrane potential.


    """
    return x

  def rev_scale_offset(self, x, bias=None, scale=None):
    """
    Reversely transform the standard range to the original membrane potential.
    """
    return x

  def clone(self):
    """
    Clone the scaling object.
    """
    return IdMemScaling()


@jax.tree_util.register_pytree_node_class
class DotDict(dict):
  """Python dictionaries with advanced dot notation access.

  For example:

  >>> d = DotDict({'a': 10, 'b': 20})
  >>> d.a
  10
  >>> d['a']
  10
  >>> d.c  # this will raise a KeyError
  KeyError: 'c'
  >>> d.c = 30  # but you can assign a value to a non-existing item
  >>> d.c
  30
  """

  __module__ = 'brainstate.util'

  def __init__(self, *args, **kwargs):
    object.__setattr__(self, '__parent', kwargs.pop('__parent', None))
    object.__setattr__(self, '__key', kwargs.pop('__key', None))
    for arg in args:
      if not arg:
        continue
      elif isinstance(arg, dict):
        for key, val in arg.items():
          self[key] = self._hook(val)
      elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
        self[arg[0]] = self._hook(arg[1])
      else:
        for key, val in iter(arg):
          self[key] = self._hook(val)

    for key, val in kwargs.items():
      self[key] = self._hook(val)

  def __setattr__(self, name, value):
    if hasattr(self.__class__, name):
      raise AttributeError(f"Attribute '{name}' is read-only in '{type(self)}' object.")
    else:
      self[name] = value

  def __setitem__(self, name, value):
    super(DotDict, self).__setitem__(name, value)
    try:
      p = object.__getattribute__(self, '__parent')
      key = object.__getattribute__(self, '__key')
    except AttributeError:
      p = None
      key = None
    if p is not None:
      p[key] = self
      object.__delattr__(self, '__parent')
      object.__delattr__(self, '__key')

  @classmethod
  def _hook(cls, item):
    if isinstance(item, dict):
      return cls(item)
    elif isinstance(item, (list, tuple)):
      return type(item)(cls._hook(elem) for elem in item)
    return item

  def __getattr__(self, item):
    return self.__getitem__(item)

  def __delattr__(self, name):
    del self[name]

  def copy(self):
    return copy.copy(self)

  def deepcopy(self):
    return copy.deepcopy(self)

  def __deepcopy__(self, memo):
    other = self.__class__()
    memo[id(self)] = other
    for key, value in self.items():
      other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
    return other

  def to_dict(self):
    base = {}
    for key, value in self.items():
      if isinstance(value, type(self)):
        base[key] = value.to_dict()
      elif isinstance(value, (list, tuple)):
        base[key] = type(value)(item.to_dict() if isinstance(item, type(self)) else item
                                for item in value)
      else:
        base[key] = value
    return base

  def update(self, *args, **kwargs):
    other = {}
    if args:
      if len(args) > 1:
        raise TypeError()
      other.update(args[0])
    other.update(kwargs)
    for k, v in other.items():
      if (k not in self) or (not isinstance(self[k], dict)) or (not isinstance(v, dict)):
        self[k] = v
      else:
        self[k].update(v)

  def __getnewargs__(self):
    return tuple(self.items())

  def __getstate__(self):
    return self

  def __setstate__(self, state):
    self.update(state)

  def setdefault(self, key, default=None):
    if key in self:
      return self[key]
    else:
      self[key] = default
      return default

  def tree_flatten(self):
    return tuple(self.values()), tuple(self.keys())

  @classmethod
  def tree_unflatten(cls, keys, values):
    return cls(jax.util.safe_zip(keys, values))


def _is_not_instance(x, cls):
  return not isinstance(x, cls)


def _is_instance(x, cls):
  return isinstance(x, cls)


@set_module_as('brainstate.util')
def not_instance_eval(*cls):
  """
  Create a partial function to evaluate if the input is not an instance of the given class.

  Args:
    *cls: The classes to check.

  Returns:
    The partial function.

  """
  return functools.partial(_is_not_instance, cls=cls)


@set_module_as('brainstate.util')
def is_instance_eval(*cls):
  """
  Create a partial function to evaluate if the input is an instance of the given class.

  Args:
    *cls: The classes to check.

  Returns:
    The partial function.
  """
  return functools.partial(_is_instance, cls=cls)
