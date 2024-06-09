``brainstate.nn`` module
========================

.. currentmodule:: brainstate.nn 
.. automodule:: brainstate.nn 

Base Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ExplicitInOutSize
   ElementWiseBlock
   Sequential
   DnnLayer


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


Connection Layers
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Linear
   ScaledWSLinear
   SignedWLinear
   CSRLinear
   Conv1d
   Conv2d
   Conv3d
   ScaledWSConv1d
   ScaledWSConv2d
   ScaledWSConv3d


Neuronal/Synaptic Dynamics
--------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Neuron
   IF
   LIF
   ALIF
   Synapse
   Expon
   STP
   STD


Rate RNNs
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   RNNCell
   ValinaRNNCell
   GRUCell
   MGUCell
   LSTMCell
   URLSTMCell


Readout Layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LeakyRateReadout
   LeakySpikeReadout


Synaptic Outputs
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SynOut
   COBA
   CUBA
   MgBlock


Element-wise Layers
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Threshold
   ReLU
   RReLU
   Hardtanh
   ReLU6
   Sigmoid
   Hardsigmoid
   Tanh
   SiLU
   Mish
   Hardswish
   ELU
   CELU
   SELU
   GLU
   GELU
   Hardshrink
   LeakyReLU
   LogSigmoid
   Softplus
   Softshrink
   PReLU
   Softsign
   Tanhshrink
   Softmin
   Softmax
   Softmax2d
   LogSoftmax
   Dropout
   Dropout1d
   Dropout2d
   Dropout3d
   AlphaDropout
   FeatureAlphaDropout
   Identity
   SpikeBitwise


Normalization Layers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BatchNorm1d
   BatchNorm2d
   BatchNorm3d


Pooling Layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Flatten
   Unflatten
   AvgPool1d
   AvgPool2d
   AvgPool3d
   MaxPool1d
   MaxPool2d
   MaxPool3d
   AdaptiveAvgPool1d
   AdaptiveAvgPool2d
   AdaptiveAvgPool3d
   AdaptiveMaxPool1d
   AdaptiveMaxPool2d
   AdaptiveMaxPool3d


Other Layers
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DropoutFixed


