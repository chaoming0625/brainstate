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

from typing import Optional, Union

from brainstate._utils import set_module_as
from brainstate._module import (Module, Dynamics, Projection, ReceiveInputProj,
                                UpdateReturn, register_delay_of_target)
from brainstate.mixin import (Mode, BindCondData)
from ._utils import is_instance

__all__ = [
  'HalfProjDelta', 'FullProjDelta',
]


class _Delta(BindCondData):
  def __init__(self):
    self._cond = None

  def bind_cond(self, cond):
    self._cond = cond

  def __call__(self, *args, **kwargs):
    r = self._cond
    return r


@set_module_as('brainstate.nn')
class HalfProjDelta(Projection):
  """Defining the half-part of the synaptic projection for the Delta synapse model.

  The synaptic projection requires the input is the spiking data, otherwise
  the synapse is not the Delta synapse model.

  The ``half-part`` means that the model only includes ``comm`` -> ``syn`` -> ``out`` -> ``post``.
  Therefore, the model's ``update`` function needs the manual providing of the spiking input.

  **Model Descriptions**

  .. math::

      I_{syn} (t) = \sum_{j\in C} g_{\mathrm{max}} * \delta(t-t_j-D)

  where :math:`g_{\mathrm{max}}` denotes the chemical synaptic strength,
  :math:`t_j` the spiking moment of the presynaptic neuron :math:`j`,
  :math:`C` the set of neurons connected to the post-synaptic neuron,
  and :math:`D` the transmission delay of chemical synapses.
  For simplicity, the rise and decay phases of post-synaptic currents are
  omitted in this model.


  **Code Examples**

  .. code-block::

      import brainstate as bp
      import brainstate.math as bm

      class Net(bp.DynamicalSystem):
        def __init__(self):
          super().__init__()

          self.pre = bp.dyn.PoissonGroup(10, 100.)
          self.post = bp.dyn.LifRef(1)
          self.syn = bp.dyn.HalfProjDelta(bp.dnn.Linear(10, 1, bp.init.OneInit(2.)), self.post)

        def update(self):
          self.syn(self.pre())
          self.post()
          return self.post.V.value

      net = Net()
      indices = bm.arange(1000).to_numpy()
      vs = bm.for_loop(net.step_run, indices, progress_bar=True)
      bp.visualize.line_plot(indices, vs, show=True)

  Args:
    comm: DynamicalSystem. The synaptic communication.
    post: DynamicalSystem. The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """

  _invisible_nodes = ['post']

  def __init__(
      self,
      comm: Module,
      post: ReceiveInputProj,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(comm, Module)
    is_instance(post, ReceiveInputProj)
    self.comm = comm

    # output initialization
    out = _Delta()
    post.add_input_fun(self.name, out, category='delta')
    self.out = out

    # references
    self.post = post

  def update(self, x):
    # call the communication
    current = self.comm(x)
    # bind the output
    self.out.bind_cond(current)
    # return the current, if needed
    return current


@set_module_as('brainstate.nn')
class FullProjDelta(Projection):
  """Full-chain of the synaptic projection for the Delta synapse model.

  The synaptic projection requires the input is the spiking data, otherwise
  the synapse is not the Delta synapse model.

  The ``full-chain`` means that the model needs to provide all information needed for a projection,
  including ``pre`` -> ``delay`` -> ``comm`` -> ``post``.

  **Model Descriptions**

  .. math::

      I_{syn} (t) = \sum_{j\in C} g_{\mathrm{max}} * \delta(t-t_j-D)

  where :math:`g_{\mathrm{max}}` denotes the chemical synaptic strength,
  :math:`t_j` the spiking moment of the presynaptic neuron :math:`j`,
  :math:`C` the set of neurons connected to the post-synaptic neuron,
  and :math:`D` the transmission delay of chemical synapses.
  For simplicity, the rise and decay phases of post-synaptic currents are
  omitted in this model.


  **Code Examples**

  .. code-block::

      import brainstate as bp
      import brainstate.math as bm


      class Net(bp.DynamicalSystem):
        def __init__(self):
          super().__init__()

          self.pre = bp.dyn.PoissonGroup(10, 100.)
          self.post = bp.dyn.LifRef(1)
          self.syn = bp.dyn.FullProjDelta(self.pre, 0., bp.dnn.Linear(10, 1, bp.init.OneInit(2.)), self.post)

        def update(self):
          self.syn()
          self.pre()
          self.post()
          return self.post.V.value


      net = Net()
      indices = bm.arange(1000).to_numpy()
      vs = bm.for_loop(net.step_run, indices, progress_bar=True)
      bp.visualize.line_plot(indices, vs, show=True)


  Args:
    pre: The pre-synaptic neuron group.
    delay: The synaptic delay.
    comm: DynamicalSystem. The synaptic communication.
    post: DynamicalSystem. The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """

  _invisible_nodes = ['pre', 'post', 'delay']

  def __init__(
      self,
      pre: UpdateReturn,
      delay: Union[None, int, float],
      comm: Module,
      post: Dynamics,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(pre, UpdateReturn)
    is_instance(comm, Module)
    is_instance(post, Dynamics)
    self.comm = comm

    # delay initialization
    if delay is not None and delay > 0.:
      delay_cls = register_delay_of_target(pre)
      delay_cls.register_entry(self.name, delay)
      self.delay = delay_cls
      self.has_delay = True
    else:
      self.delay = None
      self.has_delay = False

    # output initialization
    out = _Delta()
    post.add_input_fun(self.name, out, category='delta')
    self.out = out

    # references
    self.pre = pre
    self.post = post

  def update(self):
    # get delay
    if self.has_delay:
      x = self.delay.at(self.name)
    else:
      x = self.pre.update_return()
    # call the communication
    current = self.comm(x)
    # bind the output
    self.out.bind_cond(current)
    # return the current, if needed
    return current
