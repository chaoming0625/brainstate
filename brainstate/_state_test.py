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


import unittest

import brainstate as bc


class TestStateSourceInfo(unittest.TestCase):

  def test_state_source_info(self):
    state = bc.State(bc.random.randn(10))
    print(state._source_info)


class TestStateRepr(unittest.TestCase):

  def test_state_repr(self):
    print()

    state = bc.State(bc.random.randn(10))
    print(state)

    state2 = bc.State({'a': bc.random.randn(10), 'b': bc.random.randn(10)})
    print(state2)

    state3 = bc.State([bc.random.randn(10), bc.random.randn(10)])
    print(state3)
