#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:47:57 2022

@author: bemi


FourierFeatures


From: https://github.com/google-research/vdm 


# Copyright 2022 The VDM Authors.
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




"""


from flax import linen as nn
from jax import numpy as jnp



class Base2FourierFeatures(nn.Module):
    start: int = 0
    stop: int = 8
    step: int = 1
    
    @nn.compact
    def __call__(self, inputs):
      freqs = range(self.start, self.stop, self.step)
    
      # Create Base 2 Fourier features
      w = 2.**(jnp.asarray(freqs, dtype=inputs.dtype)) * 2 * jnp.pi
      w = jnp.tile(w[None, :], (1, inputs.shape[-1]))
    
      # Compute features
      h = jnp.repeat(inputs, len(freqs), axis=-1)
      h = w * h
      h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
      return h







