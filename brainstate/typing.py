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


from typing import Any, Sequence, Protocol, Union

import jax
import numpy as np

__all__ = [
  'Size',
  'Axes',
  'SeedOrKey',
  'ArrayLike',
  'DType',
  'DTypeLike',
]

Size = Union[int, Sequence[int]]
Axes = Union[int, Sequence[int]]
SeedOrKey = Union[int, jax.Array, np.ndarray]

# --- Array --- #

# ArrayLike is a Union of all objects that can be implicitly converted to a
# standard JAX array (i.e. not including future non-standard array types like
# KeyArray and BInt). It's different than np.typing.ArrayLike in that it doesn't
# accept arbitrary sequences, nor does it accept string data.
ArrayLike = Union[
  jax.Array,  # JAX array type
  np.ndarray,  # NumPy array type
  np.bool_, np.number,  # NumPy scalar types
  bool, int, float, complex,  # Python scalar types
]

# --- Dtype --- #


DType = np.dtype


class SupportsDType(Protocol):
  @property
  def dtype(self) -> DType: ...


# DTypeLike is meant to annotate inputs to np.dtype that return
# a valid JAX dtype. It's different than numpy.typing.DTypeLike
# because JAX doesn't support objects or structured dtypes.
# Unlike np.typing.DTypeLike, we exclude None, and instead require
# explicit annotations when None is acceptable.
DTypeLike = Union[
  str,  # like 'float32', 'int32'
  type[Any],  # like np.float32, np.int32, float, int
  np.dtype,  # like np.dtype('float32'), np.dtype('int32')
  SupportsDType,  # like jnp.float32, jnp.int32
]
