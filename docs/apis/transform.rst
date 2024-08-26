``brainstate.transform`` module
===============================

.. currentmodule:: brainstate.transform 
.. automodule:: brainstate.transform 



Gradient Transformation
-----------------------

.. autosummary::
   :toctree: generated/

   vector_grad
   grad
   jacrev
   jacfwd
   jacobian
   hessian


Condition Transformations
-------------------------

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




Mapping Transformation
----------------------

.. autosummary::
   :toctree: generated/

   map



Transformation Tools
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   StatefulFunction
   ProgressBar
   make_jaxpr
   jit_error_if
   unvmap

