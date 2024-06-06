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
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from brainstate import environ
from brainstate._utils import set_module_as


__all__ = [
  'get_dtype',
  'is_float',
  'is_int',
  'exprel',
  'flatten',
  'unflatten',
  'remove_diag',
  'clip_by_norm',
  'from_numpy',
  'as_numpy',
  'tree_zeros_like',
  'tree_ones_like',
]


@set_module_as('brainstate.math')
def get_dtype(a):
  """
  Get the dtype of a.
  """
  if hasattr(a, 'dtype'):
    return a.dtype
  else:
    if isinstance(a, bool):
      return bool
    elif isinstance(a, int):
      return environ.ditype()
    elif isinstance(a, float):
      return environ.dftype()
    elif isinstance(a, complex):
      return environ.dctype()
    else:
      raise ValueError(f'Can not get dtype of {a}.')


@set_module_as('brainstate.math')
def is_float(array):
  """
  Check if the array is a floating point array.

  Args:
    array: The input array.

  Returns:
    A boolean value indicating if the array is a floating point array.
  """
  return jnp.issubdtype(get_dtype(array), jnp.floating)


@set_module_as('brainstate.math')
def is_int(array):
  """
  Check if the array is an integer array.

  Args:
    array: The input array.

  Returns:
    A boolean value indicating if the array is an integer array.
  """
  return jnp.issubdtype(get_dtype(array), jnp.integer)


@set_module_as('brainstate.math')
def exprel(x):
  """
  Relative error exponential, ``(exp(x) - 1)/x``.

  When ``x`` is near zero, ``exp(x)`` is near 1, so the numerical calculation of ``exp(x) - 1`` can
  suffer from catastrophic loss of precision. ``exprel(x)`` is implemented to avoid the loss of
  precision that occurs when ``x`` is near zero.

  Args:
    x: ndarray. Input array. ``x`` must contain real numbers.

  Returns:
    ``(exp(x) - 1)/x``, computed element-wise.
  """

  # following the implementation of exprel from scipy.special
  x = jnp.asarray(x)
  dtype = x.dtype

  # Adjust the tolerance based on the dtype of x
  if dtype == jnp.float64:
    small_threshold = 1e-16
    big_threshold = 717
  elif dtype == jnp.float32:
    small_threshold = 1e-8
    big_threshold = 100
  elif dtype == jnp.float16:
    small_threshold = 1e-4
    big_threshold = 10
  else:
    small_threshold = 1e-4
    big_threshold = 10

  small = jnp.abs(x) < small_threshold
  big = x > big_threshold
  origin = jnp.expm1(x) / x
  return jnp.where(small, 1.0, jnp.where(big, jnp.inf, origin))


@set_module_as('brainstate.math')
def remove_diag(arr):
  """Remove the diagonal of the matrix.

  Parameters
  ----------
  arr: ArrayType
    The matrix with the shape of `(M, N)`.

  Returns
  -------
  arr: Array
    The matrix without diagonal which has the shape of `(M, N-1)`.
  """
  if arr.ndim != 2:
    raise ValueError(f'Only support 2D matrix, while we got a {arr.ndim}D array.')
  eyes = jnp.fill_diagonal(jnp.ones(arr.shape, dtype=bool), False)
  return jnp.reshape(arr[eyes], (arr.shape[0], arr.shape[1] - 1))


@set_module_as('brainstate.math')
def clip_by_norm(t, clip_norm, axis=None):
  """
  Clip the tensor by the norm of the tensor.

  Args:
    t: The tensor to be clipped.
    clip_norm: The maximum norm value.
    axis: The axis to calculate the norm. If None, the norm is calculated over the whole tensor.

  Returns:
    The clipped tensor.

  """
  return jax.tree.map(
    lambda l: l * clip_norm / jnp.maximum(jnp.sqrt(jnp.sum(l * l, axis=axis, keepdims=True)), clip_norm),
    t
  )


@set_module_as('brainstate.math')
def flatten(
    input: jax.typing.ArrayLike,
    start_dim: Optional[int] = None,
    end_dim: Optional[int] = None
) -> jax.Array:
  """Flattens input by reshaping it into a one-dimensional tensor.
  If ``start_dim`` or ``end_dim`` are passed, only dimensions starting
  with ``start_dim`` and ending with ``end_dim`` are flattened.
  The order of elements in input is unchanged.

  .. note::
     Flattening a zero-dimensional tensor will return a one-dimensional view.

  Parameters
  ----------
  input: Array
    The input array.
  start_dim: int
    the first dim to flatten
  end_dim: int
    the last dim to flatten

  Returns
  -------
  out: Array
  """
  shape = input.shape
  ndim = input.ndim
  if ndim == 0:
    ndim = 1
  if start_dim is None:
    start_dim = 0
  elif start_dim < 0:
    start_dim = ndim + start_dim
  if end_dim is None:
    end_dim = ndim - 1
  elif end_dim < 0:
    end_dim = ndim + end_dim
  end_dim += 1
  if start_dim < 0 or start_dim > ndim:
    raise ValueError(f'start_dim {start_dim} is out of size.')
  if end_dim < 0 or end_dim > ndim:
    raise ValueError(f'end_dim {end_dim} is out of size.')
  new_shape = shape[:start_dim] + (np.prod(shape[start_dim: end_dim], dtype=int),) + shape[end_dim:]
  return jnp.reshape(input, new_shape)


@set_module_as('brainstate.math')
def unflatten(x: jax.typing.ArrayLike, dim: int, sizes: Sequence[int]) -> jax.Array:
  """
  Expands a dimension of the input tensor over multiple dimensions.

  Args:
    x: input tensor.
    dim: Dimension to be unflattened, specified as an index into ``x.shape``.
    sizes: New shape of the unflattened dimension. One of its elements can be -1
        in which case the corresponding output dimension is inferred.
        Otherwise, the product of ``sizes`` must equal ``input.shape[dim]``.

  Returns:
    A tensor with the same data as ``input``, but with ``dim`` split into multiple dimensions.
    The returned tensor has one more dimension than the input tensor.
    The returned tensor shares the same underlying data with this tensor.
  """
  assert x.ndim > dim, ('The dimension to be unflattened should '
                        'be less than the tensor dimension. '
                        f'Got {dim} and {x.ndim}.')
  shape = x.shape
  new_shape = shape[:dim] + tuple(sizes) + shape[dim + 1:]
  return jnp.reshape(x, new_shape)


@set_module_as('brainstate.math')
def from_numpy(x):
  """
  Convert the numpy array to jax array.

  Args:
    x: The numpy array.

  Returns:
    The jax array.
  """
  return jnp.array(x)


@set_module_as('brainstate.math')
def as_numpy(x):
  """
  Convert the array to numpy array.

  Args:
    x: The array.

  Returns:
    The numpy array.
  """
  return np.array(x)


@set_module_as('brainstate.math')
def tree_zeros_like(tree):
  """
  Create a tree with the same structure as the input tree, but with zeros in each leaf.

  Args:
    tree: The input tree.

  Returns:
    The tree with zeros in each leaf.
  """
  return jax.tree_map(jnp.zeros_like, tree)


@set_module_as('brainstate.math')
def tree_ones_like(tree):
  """
  Create a tree with the same structure as the input tree, but with ones in each leaf.

  Args:
    tree: The input tree.

  Returns:
    The tree with ones in each leaf.

  """
  return jax.tree_map(jnp.ones_like, tree)
