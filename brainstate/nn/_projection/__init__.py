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

This module defines the basic classes for synaptic projections.

"""

from ._align_post import *
from ._align_post import __all__ as align_post_all
from ._align_pre import *
from ._align_pre import __all__ as align_pre_all
from ._delta import *
from ._delta import __all__ as delta_all
from ._vanilla import *
from ._vanilla import __all__ as vanilla_all

__all__ = align_post_all + align_pre_all + delta_all + vanilla_all
del align_post_all, align_pre_all, delta_all, vanilla_all
