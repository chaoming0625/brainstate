``brainstate.transform`` module
===============================

.. currentmodule:: brainstate.transform 
.. automodule:: brainstate.transform 



Gradient Transformations
------------------------

.. autosummary::
   :toctree: generated/

   vector_grad
   grad
   jacrev
   jacfwd
   jacobian
   hessian


Control Flow Transformations
----------------------------

.. autosummary::
   :toctree: generated/

   cond
   switch
   ifelse



For Loop Transformations
------------------------


These transformations collect the results of a loop into a single array.

.. autosummary::
   :toctree: generated/

   scan
   checkpointed_scan
   for_loop
   checkpointed_for_loop


While Loop Transformations
--------------------------

.. autosummary::
   :toctree: generated/

   while_loop
   bounded_while_loop


JIT Compilation
---------------

.. autosummary::
   :toctree: generated/

   jit


Transform Tools
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   StatefulFunction
   ProgressBar
   make_jaxpr
   jit_error_if
   unvmap

