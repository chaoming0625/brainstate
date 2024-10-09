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

# -*- coding: utf-8 -*-

from __future__ import annotations

import numbers
from typing import Callable

import jax
import jax.numpy as jnp

from ._base import DnnLayer
from ._dynamics import Neuron
from ._misc import exp_euler_step
from .. import environ, init, surrogate
from .._state import ShortTermState, ParamState
from ..mixin import Mode
from brainstate.typing import Size, ArrayLike, DTypeLike

__all__ = [
  'LeakyRateReadout',
  'LeakySpikeReadout',
]


class LeakyRateReadout(DnnLayer):
  """
  Leaky dynamics for the read-out module used in the Real-Time Recurrent Learning.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: Size,
      out_size: Size,
      tau: ArrayLike = 5.,
      w_init: Callable = init.KaimingNormal(),
      mode: Mode = None,
      name: str = None,
  ):
    super().__init__(mode=mode, name=name)

    # parameters
    self.in_size = (in_size,) if isinstance(in_size, numbers.Integral) else tuple(in_size)
    self.out_size = (out_size,) if isinstance(out_size, numbers.Integral) else tuple(out_size)
    self.tau = init.param(tau, self.in_size)
    self.decay = jnp.exp(-environ.get_dt() / self.tau)

    # weights
    self.weight = ParamState(init.param(w_init, (self.in_size[0], self.out_size[0])))

  def init_state(self, batch_size=None, **kwargs):
    self.r = ShortTermState(init.param(init.Constant(0.), self.out_size, batch_size))

  def reset_state(self, batch_size=None, **kwargs):
    self.r.value = init.param(init.Constant(0.), self.out_size, batch_size)

  def update(self, x):
    r = self.decay * self.r.value + x @ self.weight.value
    self.r.value = r
    return r


class LeakySpikeReadout(Neuron):
  """
  Integrate-and-fire neuron model.
  """

  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: Size,
      keep_size: bool = False,
      tau: ArrayLike = 5.,
      V_th: ArrayLike = 1.,
      w_init: Callable = init.KaimingNormal(),
      spk_fun: Callable = surrogate.ReluGrad(),
      spk_dtype: DTypeLike = None,
      spk_reset: str = 'soft',
      mode: Mode = None,
      name: str = None,
  ):
    super().__init__(in_size, keep_size=keep_size, name=name, mode=mode,
                     spk_fun=spk_fun, spk_dtype=spk_dtype, spk_reset=spk_reset)

    # parameters
    self.tau = init.param(tau, (self.num,))
    self.V_th = init.param(V_th, (self.num,))

    # weights
    self.weight = ParamState(init.param(w_init, (self.in_size[0], self.out_size[0])))

  def dv(self, v, t, x):
    x = self.sum_current_inputs(v, init=x)
    return (-v + x) / self.tau

  def init_state(self, batch_size, **kwargs):
    self.V = ShortTermState(init.param(init.Constant(0.), self.varshape, batch_size))

  def reset_state(self, batch_size, **kwargs):
    self.V.value = init.param(init.Constant(0.), self.varshape, batch_size)

  @property
  def spike(self):
    return self.get_spike(self.V.value)

  def get_spike(self, V):
    v_scaled = (V - self.V_th) / self.V_th
    return self.spk_fun(v_scaled)

  def update(self, x):
    # reset
    last_V = self.V.value
    last_spike = self.get_spike(last_V)
    V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
    V = last_V - V_th * last_spike
    # membrane potential
    V = exp_euler_step(self.dv, V, environ.get('t'), x @ self.weight.value)
    V = V + self.sum_delta_inputs()
    self.V.value = V
    return self.get_spike(V)
