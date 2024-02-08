#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:39:33 2022

@author: bemi


Small score network


Based on: https://github.com/google-research/vdm


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


import jax.numpy as jnp
from jaxtyping import Array
from flax import linen as nn

from typing import Tuple

from src.config_classes.nn_config import NNConfig
from src.nn_models.fourierfeatures import Base2FourierFeatures


# TODO: make number of dense layers configurable
class ScoreNetwork(nn.Module):
    config: NNConfig
    gamma_min: float
    gamma_max: float
    input_shape: Tuple[int, int, int]

    def setup(self):
        n_embd = self.config.n_embd
        num_layers = self.config.n_layer
        
        self.dense_layers = [nn.Dense(n_embd * 2) for i in range(num_layers)]
                        
        self.last_layer_dim = get_last_layer_dim(self.input_shape)        
        self.dense_last = nn.Dense(self.last_layer_dim)
        
        self.ff = Base2FourierFeatures()
    
    
    # Small score network ignores conditioning
    def __call__(self, z, gamma_t, conditioning, deterministic=True):
        
        shape = z.shape
        z_flat = z.reshape(-1, shape[1] * shape[2] * shape[3])
        
        # Normalize gamma_t
        lb = self.gamma_min
        ub = self.gamma_max
        gamma_t_norm = ((gamma_t - lb) / (ub - lb))*2-1  # ---> [-1,+1]        
        gamma_shaped = gamma_t_norm.reshape((shape[0], 1))
        
        # Concatenate normalized gamma_t as extra channel
        h = jnp.concatenate((z_flat, gamma_shaped), axis = -1)
      
        # append Fourier features
        h_ff = self.ff(h)
        h = jnp.concatenate([h, h_ff], axis=-1)
      
        # Dense layers
        for dense in self.dense_layers:
            h = nn.swish(dense(h))
          
        h = self.dense_last(h)
        
        h = h.reshape(shape)
        
        return h


def get_last_layer_dim(input_shape: Tuple[int, int, int]):
    return input_shape[0] * input_shape[1] * input_shape[2]
    


def make_into_channel(g: Array, shape: Tuple):
    g = jnp.expand_dims(g, axis = -1)
    ch = lambda x: x * jnp.ones(shape)
    return jnp.apply_along_axis(ch, axis = 1, arr = g)






