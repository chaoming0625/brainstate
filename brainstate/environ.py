# -*- coding: utf-8 -*-


import contextlib
import functools
import os
import re
from collections import defaultdict
from typing import Any, Callable

import numpy as np
from jax import config, devices, numpy as jnp
from jax._src.typing import DTypeLike

from .mixin import Mode
from .util import MemScaling, IdMemScaling

__all__ = [
  'set', 'context', 'get', 'all',
  'set_host_device_count', 'set_platform',
  'get_host_device_count', 'get_platform',
  'get_dt', 'get_mode', 'get_mem_scaling', 'get_precision',
  'tolerance', 'register_default_behavior',
  'dftype', 'ditype', 'dutype', 'dctype',
]

# Default, there are several shared arguments in the global context.
I = 'i'  # the index of the current computation.
T = 't'  # the current time of the current computation.
JIT_ERROR_CHECK = 'jit_error_check'  # whether to record the current computation.
FIT = 'fit'  # whether to fit the model.

_NOT_PROVIDE = object()
_environment_defaults = dict()  # default environment settings
_environment_contexts = defaultdict(list)  # current environment settings
_environment_functions = dict()  # environment functions


@contextlib.contextmanager
def context(**kwargs):
  r"""
  Context-manager that sets a computing environment for brain dynamics computation.

  In BrainPy, there are several basic computation settings when constructing models,
  including ``mode`` for controlling model computing behavior, ``dt`` for numerical
  integration, ``int_`` for integer precision, and ``float_`` for floating precision.
  :py:class:`~.environment`` provides a context for model construction and
  computation. In this temporal environment, models are constructed with the given
  ``mode``, ``dt``, ``int_``, etc., environment settings.

  For instance::

  >>> import brainstate as bst
  >>> with bst.environ.context(dt=0.1) as env:
  ...     dt = bst.environ.get('dt')
  ...     print(env)

  """
  if 'platform' in kwargs:
    raise ValueError('Cannot set platform in environment context. '
                     'Please use set_platform() or set() for the global setting.')
  if 'host_device_count' in kwargs:
    raise ValueError('Cannot set host_device_count in environment context. '
                     'Please use set_host_device_count() or set() for the global setting.')

  if 'precision' in kwargs:
    last_precision = get_precision()
    _set_jax_precision(kwargs['precision'])

  try:
    for k, v in kwargs.items():

      # update the current environment
      _environment_contexts[k].append(v)

      # restore the environment functions
      if k in _environment_functions:
        _environment_functions[k](v)

    # yield the current all environment information
    yield all()
  finally:

    for k, v in kwargs.items():

      # restore the current environment
      _environment_contexts[k].pop()

      # restore the environment functions
      if k in _environment_functions:
        _environment_functions[k](get(k))

    if 'precision' in kwargs:
      _set_jax_precision(last_precision)


def get(key: str, default: Any = _NOT_PROVIDE, desc: str = None):
  """
  Get one of the default computation environment.

  Returns
  -------
  item: Any
    The default computation environment.
  """
  if key == 'platform':
    return get_platform()

  if key == 'host_device_count':
    return get_host_device_count()

  if key in _environment_contexts:
    if len(_environment_contexts[key]) > 0:
      return _environment_contexts[key][-1]
  if key in _environment_defaults:
    return _environment_defaults[key]

  if default is _NOT_PROVIDE:
    if desc is not None:
      raise KeyError(
        f"'{key}' is not found in the context. \n"
        f"You can set it by `brainstate.share.context({key}=value)` "
        f"locally or `brainstate.share.set({key}=value)` globally. \n"
        f"Description: {desc}"
      )
    else:
      raise KeyError(
        f"'{key}' is not found in the context. \n"
        f"You can set it by `brainstate.share.context({key}=value)` "
        f"locally or `brainstate.share.set({key}=value)` globally."
      )
  return default


def all() -> dict:
  """
  Get all the current default computation environment.
  """
  r = dict()
  for k, v in _environment_contexts.items():
    if v:
      r[k] = v[-1]
  for k, v in _environment_defaults.items():
    if k not in r:
      r[k] = v
  return r


def get_dt():
  """Get the numerical integrator precision.

  Returns
  -------
  dt : float
      Numerical integration precision.
  """
  return get('dt')


def get_mode() -> Mode:
  """Get the default computing mode.

  References
  ----------
  mode: Mode
    The default computing mode.
  """
  return get('mode')


def get_mem_scaling() -> MemScaling:
  """Get the default computing membrane_scaling.

  Returns
  -------
  membrane_scaling: MemScaling
    The default computing membrane_scaling.
  """
  return get('mem_scaling')


def get_platform() -> str:
  """Get the computing platform.

  Returns
  -------
  platform: str
    Either 'cpu', 'gpu' or 'tpu'.
  """
  return devices()[0].platform


def get_host_device_count():
  """
  Get the number of host devices.

  Returns
  -------
  n: int
    The number of host devices.
  """
  xla_flags = os.getenv("XLA_FLAGS", "")
  match = re.search(r"--xla_force_host_platform_device_count=(\d+)", xla_flags)
  return int(match.group(1)) if match else 1


def get_precision() -> int:
  """
  Get the default precision.

  Returns
  -------
  precision: int
    The default precision.
  """
  return get('precision')


def set(
    platform: str = None,
    host_device_count: int = None,
    mem_scaling: MemScaling = None,
    precision: int = None,
    mode: Mode = None,
    **kwargs
):
  """
  Set the global default computation environment.

  Args:
    platform: str. The computing platform. Either 'cpu', 'gpu' or 'tpu'.
    host_device_count: int. The number of host devices.
    mem_scaling: MemScaling. The membrane scaling.
    precision: int. The default precision.
    mode: Mode. The computing mode.
    **kwargs: dict. Other environment settings.
  """
  if platform is not None:
    set_platform(platform)
  if host_device_count is not None:
    set_host_device_count(host_device_count)
  if mem_scaling is not None:
    assert isinstance(mem_scaling, MemScaling), 'mem_scaling must be a MemScaling instance.'
    kwargs['mem_scaling'] = mem_scaling
  if precision is not None:
    _set_jax_precision(precision)
    kwargs['precision'] = precision
  if mode is not None:
    assert isinstance(mode, Mode), 'mode must be a Mode instance.'
    kwargs['mode'] = mode

  # set default environment
  _environment_defaults.update(kwargs)

  # update the environment functions
  for k, v in kwargs.items():
    if k in _environment_functions:
      _environment_functions[k](v)


def set_host_device_count(n):
  """
  By default, XLA considers all CPU cores as one device. This utility tells XLA
  that there are `n` host (CPU) devices available to use. As a consequence, this
  allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

  .. note:: This utility only takes effect at the beginning of your program.
      Under the hood, this sets the environment variable
      `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
      `[num_device]` is the desired number of CPU devices `n`.

  .. warning:: Our understanding of the side effects of using the
      `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
      observe some strange phenomenon when using this utility, please let us
      know through our issue or forum page. More information is available in this
      `JAX issue <https://github.com/google/jax/issues/1408>`_.

  :param int n: number of devices to use.
  """
  xla_flags = os.getenv("XLA_FLAGS", "")
  xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
  os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(n)] + xla_flags)

  # update the environment functions
  if 'host_device_count' in _environment_functions:
    _environment_functions['host_device_count'](n)


def set_platform(platform: str):
  """
  Changes platform to CPU, GPU, or TPU. This utility only takes
  effect at the beginning of your program.
  """
  assert platform in ['cpu', 'gpu', 'tpu']
  config.update("jax_platform_name", platform)

  # update the environment functions
  if 'platform' in _environment_functions:
    _environment_functions['platform'](platform)


def _set_jax_precision(precision: int):
  """
  Set the default precision.

  Args:
    precision: int. The default precision.
  """
  assert precision in [64, 32, 16, 8], f'Precision must be in [64, 32, 16, 8]. But got {precision}.'
  if precision == 64:
    config.update("jax_enable_x64", True)
  else:
    config.update("jax_enable_x64", False)


@functools.lru_cache()
def _get_uint(precision: int):
  if precision == 64:
    return np.uint64
  elif precision == 32:
    return np.uint32
  elif precision == 16:
    return np.uint16
  elif precision == 8:
    return np.uint8
  else:
    raise ValueError(f'Unsupported precision: {precision}')


@functools.lru_cache()
def _get_int(precision: int):
  if precision == 64:
    return np.int64
  elif precision == 32:
    return np.int32
  elif precision == 16:
    return np.int16
  elif precision == 8:
    return np.int8
  else:
    raise ValueError(f'Unsupported precision: {precision}')


@functools.lru_cache()
def _get_float(precision: int):
  if precision == 64:
    return np.float64
  elif precision == 32:
    return np.float32
  elif precision == 16:
    return jnp.bfloat16
    # return np.float16
  else:
    raise ValueError(f'Unsupported precision: {precision}')


@functools.lru_cache()
def _get_complex(precision: int):
  if precision == 64:
    return np.complex128
  elif precision == 32:
    return np.complex64
  elif precision == 16:
    return np.complex32
  else:
    raise ValueError(f'Unsupported precision: {precision}')


def dftype() -> DTypeLike:
  """
  Default floating data type.
  """
  return _get_float(get_precision())


def ditype() -> DTypeLike:
  """
  Default integer data type.
  """
  return _get_int(get_precision())


def dutype() -> DTypeLike:
  """
  Default unsigned integer data type.
  """
  return _get_uint(get_precision())


def dctype() -> DTypeLike:
  """
  Default complex data type.
  """
  return _get_complex(get_precision())


def tolerance():
  if get_precision() == 64:
    return jnp.array(1e-12, dtype=np.float64)
  elif get_precision() == 32:
    return jnp.array(1e-5, dtype=np.float32)
  else:
    return jnp.array(1e-2, dtype=np.float16)


def register_default_behavior(key: str, behavior: Callable, replace_if_exist: bool = False):
  """
  Register a default behavior for a specific key.

  Args:
    key: str. The key to register.
    behavior: Callable. The behavior to register. It should be a callable.
    replace_if_exist: bool. Whether to replace the behavior if the key has been registered.

  """
  assert isinstance(key, str), 'key must be a string.'
  assert callable(behavior), 'behavior must be a callable.'
  if not replace_if_exist:
    assert key not in _environment_functions, f'{key} has been registered.'
  _environment_functions[key] = behavior


set(dt=0.1, precision=32, mode=Mode(), mem_scaling=IdMemScaling())

