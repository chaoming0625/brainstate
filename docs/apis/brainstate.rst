``brainstate`` module
=====================

.. currentmodule:: brainstate 
.. automodule:: brainstate 

``State`` System
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   State
   ShortTermState
   LongTermState
   ParamState


``State`` Helpers
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   StateDictManager
   visible_state_dict
   check_state_value_tree


``Module`` System
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Module
   ModuleGroup
   Sequential
   Projection
   Dynamics
   Delay
   DelayAccess


``Module`` Helpers
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   visible_module_list
   visible_module_dict
   call_order
   init_states
   reset_states
   load_states
   save_states


Synaptic Projections
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HalfProjAlignPostMg
   FullProjAlignPostMg
   HalfProjAlignPost
   FullProjAlignPost
   FullProjAlignPreSDMg
   FullProjAlignPreDSMg
   FullProjAlignPreSD
   FullProjAlignPreDS
   HalfProjDelta
   FullProjDelta
   VanillaProj


