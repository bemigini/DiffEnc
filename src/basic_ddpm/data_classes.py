#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:18:18 2022

@author: bemi


Dataclasses for model


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


import chex
import flax
from jax import numpy as jnp

from typing import Tuple


@flax.struct.dataclass
class VDMOutput:
    loss_recon: chex.Array  # [B]
    loss_klz: chex.Array  # [B]
    loss_diff: chex.Array  # [B]
    var_0: float
    var_1: float
    
    gamma_grad_min_mean_max: Tuple[float, float, float]
    
    SNR_t_min_mean_max: Tuple[float, float, float] 
    SNR_t_times_x_t_grad_norm_min_mean_max: Tuple[float, float, float]
    eps_m_eps_hat_norm_min_mean_max: Tuple[float, float, float]
    
    t_diff_loss: Tuple[chex.Array, chex.Array, chex.Array, chex.Array]
    
    non_zero_x_t_grad: int
    x_t_grad_min_mean_max: Tuple[float, float, float] = (0., 0., 0.)
    
    inv_loss: chex.Array = jnp.zeros(1)
    






