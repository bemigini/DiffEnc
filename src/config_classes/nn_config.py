#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:05:22 2023

@author: bemi


Config for NN models


"""



from typing import Callable

import flax

import jax




@flax.struct.dataclass
class NNConfig:
    m_type: str # 'small', 'unet' or 'unet_alt'
    
    with_fourier_features: bool
    with_attention: bool    
    
    n_embd: int
    n_layer: int
    num_groups_groupnorm: int
    p_dropout: float
    
    down_conv: bool
    pooling: bool
    channel_scaling: str # 'same' or 'double'
    
    non_id_init: bool
    
    kernel_init: Callable = jax.nn.initializers.normal(0.02)


