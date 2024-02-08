#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:40:53 2022

@author: bemi


Noise schedules


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
from jaxtyping import Array


from src.config_classes.ddpm_config import DDPMConfig



def constant_init(value, dtype='float32'):
  def _init(key, shape, dtype=dtype):
    return value * jnp.ones(shape, dtype)
  return _init


class NoiseSchedule_Scalar(nn.Module):
    config: DDPMConfig
      
    def setup(self):
      init_bias = self.config.gamma_min
      init_scale = self.config.gamma_max - init_bias
      self.w = self.param('w', constant_init(init_scale), (1,))
      self.b = self.param('b', constant_init(init_bias), (1,))
      
    @nn.compact
    def __call__(self, t: float) -> Array:
      return self.b + abs(self.w) * t


class NoiseSchedule_FixedLinear(nn.Module):
    config: DDPMConfig
    
    @nn.compact
    def __call__(self, t: float) -> Array:
        config = self.config
        value = config.gamma_min + (config.gamma_max-config.gamma_min) * t 
        return value * jnp.ones((1,))
    
    
class NoiseSchedule_FixedPoly10(nn.Module):
    config: DDPMConfig
    
    @nn.compact
    def __call__(self, t: float) -> Array:
        config = self.config
        g_max = config.gamma_max
        half_g_max = g_max/2
        g_min = config.gamma_min
        value = half_g_max * t**10 + (half_g_max - g_min) * t + g_min
        return value * jnp.ones((1,))
      
      
    
    





