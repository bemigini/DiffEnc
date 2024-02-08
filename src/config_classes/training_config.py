#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:39:07 2022

@author: bemi


Training config


"""


import flax
from typing import List


@flax.struct.dataclass
class TrainingConfig:
    dataset_name: str
    seed: int 
    
    num_steps_train: int
    
    batch_size: int
    
    steps_per_logging: int
    steps_per_eval: int
    steps_per_save: int
    step_saves_to_keep: List[int]
    
    profile: bool
    
    num_steps_lr_warmup: int = 100
    # num_steps_eval: int
    
    use_mult_gpus: bool = False









