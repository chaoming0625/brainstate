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

from brainstate._module import (register_delay_of_target,
                                Projection,
                                Module,
                                Dynamics,
                                ReceiveInputProj,
                                ExtendedUpdateWithBA)
from brainstate._utils import set_module_as
from brainstate.mixin import (Mode, AllOfTypes, DelayedInitializer, BindCondData, AlignPost, UpdateReturn)
from ._utils import is_instance

__all__ = [
  'HalfProjAlignPostMg', 'FullProjAlignPostMg',
  'HalfProjAlignPost', 'FullProjAlignPost',
]


def get_post_repr(out_label, syn, out):
  return f'{out_label} // {syn.identifier} // {out.identifier}'


def align_post_add_bef_update(
    out_label: str,
    syn_desc,
    out_desc,
    post: AllOfTypes[ReceiveInputProj, ExtendedUpdateWithBA],
    proj_name: str
):
  # synapse and output initialization
  _post_repr = get_post_repr(out_label, syn_desc, out_desc)
  if not post.has_before_update(_post_repr):
    syn_cls = syn_desc()
    out_cls = out_desc()

    # synapse and output initialization
    post.add_input_fun(proj_name, out_cls, label=out_label)
    post.add_before_update(_post_repr, _AlignPost(syn_cls, out_cls))
  syn = post.get_before_update(_post_repr).syn
  out = post.get_before_update(_post_repr).out
  return syn, out


class _AlignPost(Module):
  def __init__(
      self,
      syn: Module,
      out: AllOfTypes[Dynamics, BindCondData]
  ):
    super().__init__()
    self.syn = syn
    self.out = out

  def update(self, *args, **kwargs):
    self.out.bind_cond(self.syn(*args, **kwargs))


@set_module_as('brainstate.nn')
class HalfProjAlignPostMg(Projection):
  r"""
  Defining the half part of synaptic projection with the align-post reduction and the automatic synapse merging.

  The ``half-part`` means that the model only needs to provide half information needed for a projection,
  including ``comm`` -> ``syn`` -> ``out`` -> ``post``. Therefore, the model's ``update`` function needs
  the manual providing of the spiking input.

  The ``align-post`` means that the synaptic variables have the same dimension as the post-synaptic neuron group.

  The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
  parameters (such like time constants) will also share the same synaptic variables.

  All align-post projection models prefer to use the event-driven computation mode. This means that the
  ``comm`` model should be the event-driven model.

  **Code Examples**

  To define an E/I balanced network model.
  
  .. code-block:: python

    import brainstate as bp
    import brainstate.math as bm

    class EINet(bp.DynSysGroup):
      def __init__(self):
        super().__init__()
        self.N = bp.dyn.LifRef(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                               V_initializer=bp.init.Normal(-55., 2.))
        self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
        self.E = bp.dyn.HalfProjAlignPostMg(comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                         syn=bp.dyn.Expon.desc(size=4000, tau=5.),
                                         out=bp.dyn.COBA.desc(E=0.),
                                         post=self.N)
        self.I = bp.dyn.HalfProjAlignPostMg(comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
                                         syn=bp.dyn.Expon.desc(size=4000, tau=10.),
                                         out=bp.dyn.COBA.desc(E=-80.),
                                         post=self.N)

      def update(self, input):
        spk = self.delay.at('I')
        self.E(spk[:3200])
        self.I(spk[3200:])
        self.delay(self.N(input))
        return self.N.spike.value

    model = EINet()
    indices = bm.arange(1000)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=True)

  Args:
    comm: The synaptic communication.
    syn: The synaptic dynamics.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    out_label: str. The prefix of the output function.
    name: str. The projection name.
    mode:  Mode. The computing mode.
  """

  _invisible_nodes = ['syn', 'out', 'post']

  def __init__(
      self,
      comm: Module,
      syn: DelayedInitializer[AlignPost],
      out: DelayedInitializer[BindCondData],
      post: AllOfTypes[ReceiveInputProj, ExtendedUpdateWithBA],
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(syn, DelayedInitializer[AlignPost])
    is_instance(out, DelayedInitializer[BindCondData])
    is_instance(post, AllOfTypes[ReceiveInputProj, ExtendedUpdateWithBA])

    # synapse and output initialization
    syn, out = align_post_add_bef_update(out_label, syn_desc=syn, out_desc=out, post=post, proj_name=self.name)

    # references
    self.post = post
    self.syn = syn
    self.out = out
    self.comm = comm

  def update(self, x):
    current = self.comm(x)
    self.syn.align_post_input_add(current)  # synapse post current
    return current


@set_module_as('brainstate.nn')
class FullProjAlignPostMg(Projection):
  """Full-chain synaptic projection with the align-post reduction and the automatic synapse merging.

  The ``full-chain`` means that the model needs to provide all information needed for a projection,
  including ``pre`` -> ``delay`` -> ``comm`` -> ``syn`` -> ``out`` -> ``post``.

  The ``align-post`` means that the synaptic variables have the same dimension as the post-synaptic neuron group.

  The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
  parameters (such like time constants) will also share the same synaptic variables.

  All align-post projection models prefer to use the event-driven computation mode. This means that the
  ``comm`` model should be the event-driven model.

  Moreover, it's worth noting that ``FullProjAlignPostMg`` has a different updating order with all align-pre
  projection models. The updating order of align-post projections is ``spikes`` -> ``comm`` -> ``syn`` -> ``out``.
  While, the updating order of all align-pre projection models is usually ``spikes`` -> ``syn`` -> ``comm`` -> ``out``.

  **Code Examples**

  To define an E/I balanced network model.

  .. code-block:: python

      import brainstate as bp
      import brainstate.math as bm

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.FullProjAlignPostMg(pre=self.E,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                             syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                             out=bp.dyn.COBA.desc(E=0.),
                                             post=self.E)
          self.E2I = bp.dyn.FullProjAlignPostMg(pre=self.E,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                             syn=bp.dyn.Expon.desc(size=ni, tau=5.),
                                             out=bp.dyn.COBA.desc(E=0.),
                                             post=self.I)
          self.I2E = bp.dyn.FullProjAlignPostMg(pre=self.I,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                             syn=bp.dyn.Expon.desc(size=ne, tau=10.),
                                             out=bp.dyn.COBA.desc(E=-80.),
                                             post=self.E)
          self.I2I = bp.dyn.FullProjAlignPostMg(pre=self.I,
                                             delay=0.1,
                                             comm=bp.dnn.EventJitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                             syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                             out=bp.dyn.COBA.desc(E=-80.),
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
    comm: The synaptic communication.
    syn: The synaptic dynamics.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode:  Mode. The computing mode.
  """

  _invisible_nodes = ['syn', 'out', 'post', 'pre', 'delay']

  def __init__(
      self,
      pre: AllOfTypes[ExtendedUpdateWithBA, UpdateReturn],
      delay: Union[None, int, float],
      comm: Module,
      syn: DelayedInitializer[AlignPost],
      out: DelayedInitializer[BindCondData],
      post: AllOfTypes[ReceiveInputProj, ExtendedUpdateWithBA],
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(pre, AllOfTypes[ExtendedUpdateWithBA, UpdateReturn])
    is_instance(comm, Module)
    is_instance(syn, DelayedInitializer[AlignPost])
    is_instance(out, DelayedInitializer[BindCondData])
    is_instance(post, AllOfTypes[ReceiveInputProj, ExtendedUpdateWithBA])
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

    # synapse and output initialization
    syn, out = align_post_add_bef_update(out_label, syn_desc=syn, out_desc=out, post=post, proj_name=self.name)

    # references
    self.pre = pre
    self.post = post
    self.syn = syn
    self.out = out

  def update(self):
    if self.has_delay:
      x = self.delay.at(self.name)
    else:
      x = self.pre.update_return()
    current = self.comm(x)
    self.syn.align_post_input_add(current)  # synapse post current
    return current


@set_module_as('brainstate.nn')
class HalfProjAlignPost(Projection):
  """Defining the half-part of synaptic projection with the align-post reduction.

  The ``half-part`` means that the model only needs to provide half information needed for a projection,
  including ``comm`` -> ``syn`` -> ``out`` -> ``post``. Therefore, the model's ``update`` function needs
  the manual providing of the spiking input.

  The ``align-post`` means that the synaptic variables have the same dimension as the post-synaptic neuron group.

  All align-post projection models prefer to use the event-driven computation mode. This means that the
  ``comm`` model should be the event-driven model.

  To simulate an E/I balanced network:

  .. code-block::

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          self.N = bp.dyn.LifRef(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
          self.E = bp.dyn.HalfProjAlignPost(comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                         syn=bp.dyn.Expon(size=4000, tau=5.),
                                         out=bp.dyn.COBA(E=0.),
                                         post=self.N)
          self.I = bp.dyn.HalfProjAlignPost(comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
                                         syn=bp.dyn.Expon(size=4000, tau=10.),
                                         out=bp.dyn.COBA(E=-80.),
                                         post=self.N)

        def update(self, input):
          spk = self.delay.at('I')
          self.E(spk[:3200])
          self.I(spk[3200:])
          self.delay(self.N(input))
          return self.N.spike.value

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


  Args:
    comm: The synaptic communication.
    syn: The synaptic dynamics.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode:  Mode. The computing mode.
  """

  _invisible_nodes = ['out', 'post']

  def __init__(
      self,
      comm: Module,
      syn: AlignPost,
      out: BindCondData,
      post: ReceiveInputProj,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(comm, Module)
    is_instance(syn, AlignPost)
    is_instance(out, BindCondData)
    is_instance(post, Module)
    self.comm = comm
    self.syn = syn

    # synapse and output initialization
    post.add_input_fun(self.name, out, label=out_label)

    # reference
    self.post = post
    self.out = out

  def update(self, x):
    current = self.comm(x)
    g = self.syn(current)
    self.out.bind_cond(g)  # synapse post current
    return current


@set_module_as('brainstate.nn')
class FullProjAlignPost(Projection):
  """Full-chain synaptic projection with the align-post reduction.

  The ``full-chain`` means that the model needs to provide all information needed for a projection,
  including ``pre`` -> ``delay`` -> ``comm`` -> ``syn`` -> ``out`` -> ``post``.

  The ``align-post`` means that the synaptic variables have the same dimension as the post-synaptic neuron group.

  All align-post projection models prefer to use the event-driven computation mode. This means that the
  ``comm`` model should be the event-driven model.

  Moreover, it's worth noting that ``FullProjAlignPost`` has a different updating order with all align-pre
  projection models. The updating order of align-post projections is ``spikes`` -> ``comm`` -> ``syn`` -> ``out``.
  While, the updating order of all align-pre projection models is usually ``spikes`` -> ``syn`` -> ``comm`` -> ``out``.

  To simulate and define an E/I balanced network model:

  .. code-block:: python

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          ne, ni = 3200, 800
          self.E = bp.dyn.LifRef(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.I = bp.dyn.LifRef(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.E2E = bp.dyn.FullProjAlignPost(pre=self.E,
                                           delay=0.1,
                                           comm=bp.dnn.EventJitFPHomoLinear(ne, ne, prob=0.02, weight=0.6),
                                           syn=bp.dyn.Expon(size=ne, tau=5.),
                                           out=bp.dyn.COBA(E=0.),
                                           post=self.E)
          self.E2I = bp.dyn.FullProjAlignPost(pre=self.E,
                                           delay=0.1,
                                           comm=bp.dnn.EventJitFPHomoLinear(ne, ni, prob=0.02, weight=0.6),
                                           syn=bp.dyn.Expon(size=ni, tau=5.),
                                           out=bp.dyn.COBA(E=0.),
                                           post=self.I)
          self.I2E = bp.dyn.FullProjAlignPost(pre=self.I,
                                           delay=0.1,
                                           comm=bp.dnn.EventJitFPHomoLinear(ni, ne, prob=0.02, weight=6.7),
                                           syn=bp.dyn.Expon(size=ne, tau=10.),
                                           out=bp.dyn.COBA(E=-80.),
                                           post=self.E)
          self.I2I = bp.dyn.FullProjAlignPost(pre=self.I,
                                           delay=0.1,
                                           comm=bp.dnn.EventJitFPHomoLinear(ni, ni, prob=0.02, weight=6.7),
                                           syn=bp.dyn.Expon(size=ni, tau=10.),
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
    comm: The synaptic communication.
    syn: The synaptic dynamics.
    out: The synaptic output.
    post: The post-synaptic neuron group.
    name: str. The projection name.
    mode:  Mode. The computing mode.
  """

  _invisible_nodes = ['post', 'pre', 'delay']

  def __init__(
      self,
      pre: UpdateReturn,
      delay: Union[None, int, float],
      comm: Module,
      syn: AlignPost,
      out: BindCondData,
      post: ReceiveInputProj,
      out_label: Optional[str] = None,
      name: Optional[str] = None,
      mode: Optional[Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    is_instance(pre, UpdateReturn)
    is_instance(comm, Module)
    is_instance(syn, AllOfTypes[Dynamics, AlignPost])
    is_instance(out, AllOfTypes[Dynamics, BindCondData])
    is_instance(post, ReceiveInputProj)
    self.comm = comm
    self.syn = syn
    self.out = out

    # delay initialization
    if delay is not None and delay > 0.:
      delay_cls = register_delay_of_target(pre)
      delay_cls.register_entry(self.name, delay)
      self.delay = delay_cls
      self.has_delay = True
    else:
      self.delay = None
      self.has_delay = False

    # synapse and output initialization
    post.add_input_fun(self.name, out, label=out_label)

    # references
    self.post = post
    self.pre = pre

  def update(self):
    if self.has_delay:
      x = self.delay.at(self.name)
    else:
      x = self.pre.update_return()
    g = self.syn(self.comm(x))
    self.out.bind_cond(g)  # synapse post current
    return g
