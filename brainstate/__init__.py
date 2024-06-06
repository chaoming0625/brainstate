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

"""
A ``State``-based Transformation System for Brain Dynamics Programming
"""

__version__ = "0.0.1"

from . import environ
from . import math
from . import mixin
from . import nn
from . import optim
from . import random
from . import transform
from . import typing
from . import util
from ._module import *
from ._module import __all__ as _module_all
from ._state import *
from ._state import __all__ as _state_all

__all__ = (
    ['environ', 'share', 'nn', 'optim', 'random',
     'mixin', 'math', 'transform', 'util', 'typing'] +
    _module_all + _state_all
)
del _module_all, _state_all
