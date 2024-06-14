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

import jax.numpy as jnp

import brainstate as bc


class TestVarDelay(unittest.TestCase):
  def test_delay1(self):
    a = bc.State(bc.random.random(10, 20))
    delay = bc.Delay(a.value)
    delay.register_entry('a', 1.)
    delay.register_entry('b', 2.)
    delay.register_entry('c', None)

    delay.init_state()
    with self.assertRaises(KeyError):
      delay.register_entry('c', 10.)
    bc.util.clear_buffer_memory()

  def test_rotation_delay(self):
    rotation_delay = bc.Delay(jnp.ones((1,)))
    t0 = 0.
    t1, n1 = 1., 10
    t2, n2 = 2., 20

    rotation_delay.register_entry('a', t0)
    rotation_delay.register_entry('b', t1)
    rotation_delay.register_entry('c2', 1.9)
    rotation_delay.register_entry('c', t2)

    rotation_delay.init_state()

    print()
    # print(rotation_delay)
    # print(rotation_delay.max_length)

    for i in range(100):
      bc.environ.set(i=i)
      rotation_delay(jnp.ones((1,)) * i)
      # print(i, rotation_delay.at('a'), rotation_delay.at('b'), rotation_delay.at('c2'), rotation_delay.at('c'))
      self.assertTrue(jnp.allclose(rotation_delay.at('a'), jnp.ones((1,)) * i))
      self.assertTrue(jnp.allclose(rotation_delay.at('b'), jnp.maximum(jnp.ones((1,)) * i - n1, 0.)))
      self.assertTrue(jnp.allclose(rotation_delay.at('c'), jnp.maximum(jnp.ones((1,)) * i - n2, 0.)))
    bc.util.clear_buffer_memory()

  def test_concat_delay(self):
    rotation_delay = bc.Delay(jnp.ones([1]), method='concat')
    t0 = 0.
    t1, n1 = 1., 10
    t2, n2 = 2., 20

    rotation_delay.register_entry('a', t0)
    rotation_delay.register_entry('b', t1)
    rotation_delay.register_entry('c', t2)

    rotation_delay.init_state()

    print()
    for i in range(100):
      bc.environ.set(i=i)
      rotation_delay(jnp.ones((1,)) * i)
      print(i, rotation_delay.at('a'), rotation_delay.at('b'), rotation_delay.at('c'))
      self.assertTrue(jnp.allclose(rotation_delay.at('a'), jnp.ones((1,)) * i))
      self.assertTrue(jnp.allclose(rotation_delay.at('b'), jnp.maximum(jnp.ones((1,)) * i - n1, 0.)))
      self.assertTrue(jnp.allclose(rotation_delay.at('c'), jnp.maximum(jnp.ones((1,)) * i - n2, 0.)))
    bc.util.clear_buffer_memory()

  def test_rotation_and_concat_delay(self):
    rotation_delay = bc.Delay(jnp.ones((1,)))
    concat_delay = bc.Delay(jnp.ones([1]), method='concat')
    t0 = 0.
    t1, n1 = 1., 10
    t2, n2 = 2., 20

    rotation_delay.register_entry('a', t0)
    rotation_delay.register_entry('b', t1)
    rotation_delay.register_entry('c', t2)
    concat_delay.register_entry('a', t0)
    concat_delay.register_entry('b', t1)
    concat_delay.register_entry('c', t2)

    rotation_delay.init_state()
    concat_delay.init_state()

    print()
    for i in range(100):
      bc.environ.set(i=i)
      new = jnp.ones((1,)) * i
      rotation_delay(new)
      concat_delay(new)
      self.assertTrue(jnp.allclose(rotation_delay.at('a'), concat_delay.at('a'), ))
      self.assertTrue(jnp.allclose(rotation_delay.at('b'), concat_delay.at('b'), ))
      self.assertTrue(jnp.allclose(rotation_delay.at('c'), concat_delay.at('c'), ))
    bc.util.clear_buffer_memory()


class TestModule(unittest.TestCase):
  def test_states(self):
    class A(bc.Module):
      def __init__(self):
        super().__init__()
        self.a = bc.State(bc.random.random(10, 20))
        self.b = bc.State(bc.random.random(10, 20))

    class B(bc.Module):
      def __init__(self):
        super().__init__()
        self.a = A()
        self.b = bc.State(bc.random.random(10, 20))

    b = B()
    print()
    print(b.states())
    print(b.states())
    print(b.states(level=0))
    print(b.states(level=0))
