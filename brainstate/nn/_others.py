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

from functools import partial
from typing import Optional

import jax.numpy as jnp

from . import init
from ._base import DnnLayer
from .. import random, math, environ, typing
from ..mixin import Mode

__all__ = [
  'DropoutFixed',
]


class DropoutFixed(DnnLayer):
  """
  A dropout layer with the fixed dropout mask along the time axis once after initialized.

  In training, to compensate for the fraction of input values dropped (`rate`),
  all surviving values are multiplied by `1 / (1 - rate)`.

  This layer is active only during training (``mode=brainstate.mixin.Training``). In other
  circumstances it is a no-op.

  .. [1] Srivastava, Nitish, et al. "Dropout: a simple way to prevent
         neural networks from overfitting." The journal of machine learning
         research 15.1 (2014): 1929-1958.

  .. admonition:: Tip
      :class: tip

      This kind of Dropout is firstly described in `Enabling Spike-based Backpropagation for Training Deep Neural
      Network Architectures <https://arxiv.org/abs/1903.06379>`_:

      There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
      training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
      are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
      iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
      the output error and modify the network parameters only at the last time step. For dropout to be effective in
      our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
      data is not changed, such that the neural network is constituted by the same random subset of units during
      each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
      each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
      iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
      are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
      time window within an iteration.

  Args:
    in_size: The size of the input tensor.
    prob: Probability to keep element of the tensor.
    mode: Mode. The computation mode of the object.
    name: str. The name of the dynamic system.
  """
  __module__ = 'brainstate.nn'

  def __init__(
      self,
      in_size: typing.Size,
      prob: float = 0.5,
      mode: Optional[Mode] = None,
      name: Optional[str] = None
  ) -> None:
    super().__init__(mode=mode, name=name)
    assert 0. <= prob < 1., f"Dropout probability must be in the range [0, 1). But got {prob}."
    self.prob = prob
    self.in_size = in_size
    self.out_size = in_size

  def init_state(self, batch_size=None, **kwargs):
    self.mask = init.param(partial(random.bernoulli, self.prob), self.in_size, batch_size)

  def update(self, x):
    dtype = math.get_dtype(x)
    fit_phase = environ.get('fit', desc='Whether this is a fitting process. Bool.')
    if fit_phase:
      assert self.mask.shape == x.shape, (f"Input shape {x.shape} does not match the mask shape {self.mask.shape}. "
                                          f"Please call `init_state()` method first.")
      return jnp.where(self.mask,
                       jnp.asarray(x / self.prob, dtype=dtype),
                       jnp.asarray(0., dtype=dtype))
    else:
      return x
