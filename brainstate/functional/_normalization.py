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

from typing import Optional

import jax
import jax.numpy as jnp

from .._utils import set_module_as

__all__ = [
  'weight_standardization',
]


@set_module_as('brainstate.functional')
def weight_standardization(
    w: jax.typing.ArrayLike,
    eps: float = 1e-4,
    gain: Optional[jax.Array] = None,
    out_axis: int = -1,
):
  """
  Scaled Weight Standardization,
  see `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization <https://paperswithcode.com/paper/weight-standardization>`_.

  Parameters
  ----------
  w : jax.typing.ArrayLike
      The weight tensor.
  eps : float
      A small value to avoid division by zero.
  gain : Array
      The gain function, by default None.
  out_axis : int
      The output axis, by default -1.

  Returns
  -------
  jax.typing.ArrayLike
      The scaled weight tensor.
  """
  if out_axis < 0:
    out_axis = w.ndim + out_axis
  fan_in = 1  # get the fan-in of the weight tensor
  axes = []  # get the axes of the weight tensor
  for i in range(w.ndim):
    if i != out_axis:
      fan_in *= w.shape[i]
      axes.append(i)
  # normalize the weight
  mean = jnp.mean(w, axis=axes, keepdims=True)
  var = jnp.var(w, axis=axes, keepdims=True)
  scale = jax.lax.rsqrt(jnp.maximum(var * fan_in, eps))
  if gain is not None:
    scale = gain * scale
  shift = mean * scale
  return w * scale - shift
