``brainstate`` documentation
============================

`brainstate <https://github.com/brainpy/brainstate>`_ implements a ``State``-based transformation system for brain dynamics programming (BDP).

----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainstate[cpu]

    .. tab-item:: GPU (CUDA 11.0)

       .. code-block:: bash

          pip install -U brainstate[cuda11]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainstate[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainstate[tpu]


----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


- `brainpy <https://github.com/brainpy/BrainPy>`_: The solution for the general-purpose brain dynamics programming.

- `brainstate <https://github.com/brainpy/brainstate>`_: The ``State``-based transformation system for brain dynamics programming.

- `braintools <https://github.com/brainpy/braintools>`_: The tools for the brain dynamics simulation and analysis.

- `brainscale <https://github.com/brainpy/brainscale>`_: The scalable online learning for biological spiking neural networks.



.. toctree::
   :hidden:
   :maxdepth: 2

   api.rst

