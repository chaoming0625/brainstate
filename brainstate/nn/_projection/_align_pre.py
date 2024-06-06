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
from brainstate._module import (Module, DelayAccess, Projection,
                                ExtendedUpdateWithBA, ReceiveInputProj,
                                register_delay_of_target)
from brainstate.mixin import (DelayedInitializer, BindCondData, UpdateReturn, Mode, AllOfTypes)
from ._utils import is_instance

__all__ = [
  'FullProjAlignPreSDMg', 'FullProjAlignPreDSMg',
  'FullProjAlignPreSD', 'FullProjAlignPreDS',
]


def align_pre_add_bef_update(
    syn_desc: DelayedInitializer,
    delay_at,
    delay_cls: ExtendedUpdateWithBA,
    proj_name: str = None
):
  _syn_id = f'Delay({str(delay_at)}) // {syn_desc.identifier}'
  if not delay_cls.has_before_update(_syn_id):
    # delay
    delay_access = DelayAccess(delay_cls, delay_at, delay_entry=proj_name)
    # synapse
    syn_cls = syn_desc()
    # add to "after_updates"
    delay_cls.add_before_update(_syn_id, _AlignPreMg(delay_access, syn_cls))
  syn = delay_cls.get_before_update(_syn_id).syn
  return syn


class _AlignPreMg(Module):
  def __init__(self, access, syn):
    super().__init__()
    self.access = access
    self.syn = syn

  def update(self, *args, **kwargs):
    return self.syn(self.access())


@set_module_as('brainstate.nn')
class FullProjAlignPreSDMg(Projection):
  """Full-chain synaptic projection with the align-pre reduction and synapse+delay updating and merging.

  The ``full-chain`` means that the model needs to provide all information needed for a projection,
  including ``pre`` -> ``syn`` -> ``delay`` -> ``comm`` -> ``out`` -> ``post``.

  The ``align-pre`` means that the synaptic variables have the same dimension as the pre-synaptic neuron group.

  The ``synapse+delay updating`` means that the projection first computes the synapse states, then delivers the
  synapse states to the delay model, and finally computes the synaptic current.

  The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
  parameters (such like time constants) will also share the same synaptic variables.

  Neither ``FullProjAlignPreSDMg`` nor ``FullProjAlignPreDSMg`` facilitates the event-driven computation.
  This is because the ``comm`` is computed after the synapse state, which is a floating-point number, rather
  than the spiking. To facilitate the event-driven computation, please use align post projections.

  To simulate an E/I balanced network model:

  .. code-block:: python

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.FullProjAlignPreSDMg(pre=self.E,
                                                syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                delay=0.1,
                                                comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.E)
          self.E2I = bp.dyn.FullProjAlignPreSDMg(pre=self.E,
                                                syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                delay=0.1,
                                                comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.I)
          self.I2E = bp.dyn.FullProjAlignPreSDMg(pre=self.I,
                                                syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                delay=0.1,
                                                comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                                out=bp.dyn.COBA(E=-80.),
                                                post=self.E)
          self.I2I = bp.dyn.FullProjAlignPreSDMg(pre=self.I,
                                                syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                delay=0.1,
                                                comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                                out=bp.dyn.COBA(E=-80.),
                                                post=self.I)

        def update(self, inp):
          self.E2E()
          self.E2I()
          self.I2E()
          self.I2I()
          self.E(inp)
          self.I(inp)
          return self.E.spike

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


  Args:
    pre: The pre-synaptic neuron group.
    syn: The synaptic dynamics.
    delay: The synaptic delay.
    comm: The synaptic communication.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """

  _invisible_nodes = ['pre', 'syn', 'delay', 'post']

  def __init__(
      self,
      pre: ExtendedUpdateWithBA,
      syn: DelayedInitializer[UpdateReturn],
      delay: Union[None, int, float],
      comm: Module,
      out: BindCondData,
      post: ReceiveInputProj,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(pre, ExtendedUpdateWithBA)
    is_instance(syn, DelayedInitializer[UpdateReturn])
    is_instance(comm, Module)
    is_instance(out, BindCondData)
    is_instance(post, ReceiveInputProj)
    self.comm = comm
    self.out = out

    # synapse initialization
    if not pre.has_after_update(syn.identifier):
      syn_cls = syn()
      pre.add_after_update(syn.identifier, syn_cls)
    self.syn = pre.get_after_update(syn.identifier)

    # delay initialization
    if delay is not None and delay > 0.:
      delay_cls = register_delay_of_target(self.syn)
      self.has_delay = True
      self.delay = delay_cls
    else:
      self.has_delay = False
      self.delay = None

    # output initialization
    post.add_input_fun(self.name, out, label=out_label)

    # references
    self.pre = pre
    self.post = post

  def update(self, x=None):
    if x is None:
      if self.has_delay:
        x = self.delay.at(self.name)
      else:
        x = self.syn.update_return()
    current = self.comm(x)
    self.out.bind_cond(current)
    return current


@set_module_as('brainstate.nn')
class FullProjAlignPreDSMg(Projection):
  """Full-chain synaptic projection with the align-pre reduction and delay+synapse updating and merging.

  The ``full-chain`` means that the model needs to provide all information needed for a projection,
  including ``pre`` -> ``delay`` -> ``syn`` -> ``comm`` -> ``out`` -> ``post``.
  Note here, compared to ``FullProjAlignPreSDMg``, the ``delay`` and ``syn`` are exchanged.

  The ``align-pre`` means that the synaptic variables have the same dimension as the pre-synaptic neuron group.

  The ``delay+synapse updating`` means that the projection first delivers the pre neuron output (usually the
  spiking)  to the delay model, then computes the synapse states, and finally computes the synaptic current.

  The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
  parameters (such like time constants) will also share the same synaptic variables.

  Neither ``FullProjAlignPreDSMg`` nor ``FullProjAlignPreSDMg`` facilitates the event-driven computation.
  This is because the ``comm`` is computed after the synapse state, which is a floating-point number, rather
  than the spiking. To facilitate the event-driven computation, please use align post projections.


  To simulate an E/I balanced network model:

  .. code-block:: python

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.FullProjAlignPreDSMg(pre=self.E,
                                                delay=0.1,
                                                syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.E)
          self.E2I = bp.dyn.FullProjAlignPreDSMg(pre=self.E,
                                                delay=0.1,
                                                syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.I)
          self.I2E = bp.dyn.FullProjAlignPreDSMg(pre=self.I,
                                                delay=0.1,
                                                syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                                out=bp.dyn.COBA(E=-80.),
                                                post=self.E)
          self.I2I = bp.dyn.FullProjAlignPreDSMg(pre=self.I,
                                                delay=0.1,
                                                syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                                out=bp.dyn.COBA(E=-80.),
                                                post=self.I)

        def update(self, inp):
          self.E2E()
          self.E2I()
          self.I2E()
          self.I2I()
          self.E(inp)
          self.I(inp)
          return self.E.spike

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


  Args:
    pre: The pre-synaptic neuron group.
    delay: The synaptic delay.
    syn: The synaptic dynamics.
    comm: The synaptic communication.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """
  _invisible_nodes = ['pre', 'syn', 'delay', 'post']

  def __init__(
      self,
      pre: AllOfTypes[ExtendedUpdateWithBA, UpdateReturn],
      delay: Union[None, int, float],
      syn: DelayedInitializer[UpdateReturn],
      comm: Module,
      out: BindCondData,
      post: ReceiveInputProj,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(pre, AllOfTypes[ExtendedUpdateWithBA, UpdateReturn])
    is_instance(syn, DelayedInitializer[Module])
    is_instance(comm, Module)
    is_instance(out, BindCondData)
    is_instance(post, ReceiveInputProj)
    self.comm = comm
    self.out = out

    # delay initialization
    if delay is not None and delay > 0.:
      delay_cls = register_delay_of_target(pre)
      self.has_delay = True
      self.delay = delay_cls
      # synapse initialization
      self.syn = align_pre_add_bef_update(syn, delay, delay_cls, self.name)
    else:
      self.has_delay = False
      self.delay = None
      if not pre.has_after_update(syn.identifier):
        syn_cls = syn()
        pre.add_after_update(syn.identifier, syn_cls)
      self.syn = pre.get_after_update(syn.identifier)

    # output initialization
    post.add_input_fun(self.name, out, label=out_label)

    # references
    self.pre = pre
    self.post = post

  def update(self):
    x = self.syn.update_return()
    current = self.comm(x)
    self.out.bind_cond(current)
    return current


@set_module_as('brainstate.nn')
class FullProjAlignPreSD(Projection):
  """Full-chain synaptic projection with the align-pre reduction and synapse+delay updating.

  The ``full-chain`` means that the model needs to provide all information needed for a projection,
  including ``pre`` -> ``syn`` -> ``delay`` -> ``comm`` -> ``out`` -> ``post``.

  The ``align-pre`` means that the synaptic variables have the same dimension as the pre-synaptic neuron group.

  The ``synapse+delay updating`` means that the projection first computes the synapse states, then delivers the
  synapse states to the delay model, and finally computes the synaptic current.

  Neither ``FullProjAlignPreSD`` nor ``FullProjAlignPreDS`` facilitates the event-driven computation.
  This is because the ``comm`` is computed after the synapse state, which is a floating-point number, rather
  than the spiking. To facilitate the event-driven computation, please use align post projections.


  To simulate an E/I balanced network model:

  .. code-block:: python

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.FullProjAlignPreSD(pre=self.E,
                                              syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                              delay=0.1,
                                              comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                              out=bp.dyn.COBA(E=0.),
                                              post=self.E)
          self.E2I = bp.dyn.FullProjAlignPreSD(pre=self.E,
                                              syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                              delay=0.1,
                                              comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                              out=bp.dyn.COBA(E=0.),
                                              post=self.I)
          self.I2E = bp.dyn.FullProjAlignPreSD(pre=self.I,
                                              syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                              delay=0.1,
                                              comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                              out=bp.dyn.COBA(E=-80.),
                                              post=self.E)
          self.I2I = bp.dyn.FullProjAlignPreSD(pre=self.I,
                                              syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                              delay=0.1,
                                              comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                              out=bp.dyn.COBA(E=-80.),
                                              post=self.I)

        def update(self, inp):
          self.E2E()
          self.E2I()
          self.I2E()
          self.I2I()
          self.E(inp)
          self.I(inp)
          return self.E.spike

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


  Args:
    pre: The pre-synaptic neuron group.
    syn: The synaptic dynamics.
    delay: The synaptic delay.
    comm: The synaptic communication.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """

  _invisible_nodes = ['pre', 'post']

  def __init__(
      self,
      pre: ExtendedUpdateWithBA,
      syn: UpdateReturn,
      delay: Union[None, int, float],
      comm: Module,
      out: BindCondData,
      post: ReceiveInputProj,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(pre, ExtendedUpdateWithBA)
    is_instance(syn, UpdateReturn)
    is_instance(comm, Module)
    is_instance(out, BindCondData)
    is_instance(post, ReceiveInputProj)
    self.comm = comm
    self.syn = syn
    self.out = out

    # delay initialization
    if delay is not None and delay > 0.:
      delay_cls = register_delay_of_target(syn)
      delay_cls.register_entry(self.name, delay)
      self.delay = delay_cls
    else:
      self.delay = None

    # output initialization
    post.add_input_fun(self.name, out, label=out_label)

    # references
    self.pre = pre
    self.post = post

  def update(self):
    if self.delay is not None:
      self.delay(self.syn(self.pre.update_return()))
      x = self.delay.at(self.name)
    else:
      x = self.syn(self.pre.update_return())
    current = self.comm(x)
    self.out.bind_cond(current)
    return current


@set_module_as('brainstate.nn')
class FullProjAlignPreDS(Projection):
  """Full-chain synaptic projection with the align-pre reduction and delay+synapse updating.

  The ``full-chain`` means that the model needs to provide all information needed for a projection,
  including ``pre`` -> ``syn`` -> ``delay`` -> ``comm`` -> ``out`` -> ``post``.
  Note here, compared to ``FullProjAlignPreSD``, the ``delay`` and ``syn`` are exchanged.

  The ``align-pre`` means that the synaptic variables have the same dimension as the pre-synaptic neuron group.

  The ``delay+synapse updating`` means that the projection first delivers the pre neuron output (usually the
  spiking)  to the delay model, then computes the synapse states, and finally computes the synaptic current.

  Neither ``FullProjAlignPreDS`` nor ``FullProjAlignPreSD`` facilitates the event-driven computation.
  This is because the ``comm`` is computed after the synapse state, which is a floating-point number, rather
  than the spiking. To facilitate the event-driven computation, please use align post projections.


  To simulate an E/I balanced network model:

  .. code-block:: python

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.FullProjAlignPreDS(pre=self.E,
                                              delay=0.1,
                                              syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                              comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                              out=bp.dyn.COBA(E=0.),
                                              post=self.E)
          self.E2I = bp.dyn.FullProjAlignPreDS(pre=self.E,
                                              delay=0.1,
                                              syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                              comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                              out=bp.dyn.COBA(E=0.),
                                              post=self.I)
          self.I2E = bp.dyn.FullProjAlignPreDS(pre=self.I,
                                              delay=0.1,
                                              syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                              comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                              out=bp.dyn.COBA(E=-80.),
                                              post=self.E)
          self.I2I = bp.dyn.FullProjAlignPreDS(pre=self.I,
                                              delay=0.1,
                                              syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                              comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                              out=bp.dyn.COBA(E=-80.),
                                              post=self.I)

        def update(self, inp):
          self.E2E()
          self.E2I()
          self.I2E()
          self.I2I()
          self.E(inp)
          self.I(inp)
          return self.E.spike

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


  Args:
    pre: The pre-synaptic neuron group.
    delay: The synaptic delay.
    syn: The synaptic dynamics.
    comm: The synaptic communication.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """

  _invisible_nodes = ['pre', 'post', 'delay']

  def __init__(
      self,
      pre: UpdateReturn,
      delay: Union[None, int, float],
      syn: Module,
      comm: Module,
      out: BindCondData,
      post: ReceiveInputProj,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(pre, UpdateReturn)
    is_instance(syn, Module)
    is_instance(comm, Module)
    is_instance(out, BindCondData)
    is_instance(post, ReceiveInputProj)
    self.comm = comm
    self.syn = syn
    self.out = out

    # delay initialization
    if delay is not None and delay > 0.:
      delay_cls = register_delay_of_target(pre)
      delay_cls.register_entry(self.name, delay)
      self.delay = delay_cls
    else:
      self.delay = None

    # output initialization
    post.add_input_fun(self.name, out, label=out_label)

    # references
    self.pre = pre
    self.post = post

  def update(self, x=None):
    if x is None:
      if self.delay is not None:
        x = self.delay.at(self.name)
      else:
        x = self.pre.update_return()
    g = self.comm(self.syn(x))
    self.out.bind_cond(g)
    return g
