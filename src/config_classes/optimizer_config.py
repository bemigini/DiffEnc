#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 09:37:23 2022

@author: bemi


Optimizer config


"""



import flax



@flax.struct.dataclass
class OptimizerArgs:
    b1: float 
    b2: float 
    eps: float 
    weight_decay: float


@flax.struct.dataclass
class OptimizerConfig:
    name: str
    args: OptimizerArgs
    learning_rate: float 
    use_gradient_clipping: bool
    gradient_clip_norm: float
    lr_decay: bool = False 
    ema_rate: float = 0.9999 # exponential moving average
    
    
    



