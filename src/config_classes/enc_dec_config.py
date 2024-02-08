#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:01:27 2022

@author: bemi


Encoder/decoder config class



"""



import flax

from src.config_classes.nn_config import NNConfig


@flax.struct.dataclass
class EncDecConfig:
    """Encoder configurations."""
    
    enc_type: str # simple or trainable
    vocab_size: int    
    
    nn_config: NNConfig
    
    id_at_zero: bool
    m1_to_1: bool
    
    k: float = 1.
    init_var: float = 0.
    min_var: float = 0.
    use_x_var: bool = False
    gamma_reg: bool = False
    end_reg: bool = False
    end_half_reg: bool = False






