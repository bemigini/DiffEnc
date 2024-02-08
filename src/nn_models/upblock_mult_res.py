#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:18:53 2023

@author: bemi


Up block


"""



from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Array

from src.nn_models.nn_blocks import ResnetBlock



class UpBlock(nn.Module):
    """
    A helper Module with 1 transposed convolution block and 1 ResNet block.
    """
    block_name: str
    features: int
    scale_up: bool = True
    p_dropout: float = 0.1
    num_res_blocks: int = 1
    
    def setup(self):        
        self.res_blocks = [ResnetBlock(
            p_dropout=self.p_dropout, 
            out_ch=self.features, 
            name=f'{self.block_name}_resnet_block{i}')
            for i in range(self.num_res_blocks)]
        
        self.conv_trans = nn.ConvTranspose(self.features, (2,2), (2,2))
            
    
    @nn.compact
    def __call__(self, x: Array, skip_layer: Array, cond: Array, deterministic: bool):
        
        if self.scale_up:
            x = self.conv_trans(x)
        
        merged = jnp.concatenate([x, skip_layer], -1)
        
        h = merged
        
        for res in self.res_blocks:
            h = res(h, cond = cond, deterministic = deterministic)[0]
               
        return h
    



