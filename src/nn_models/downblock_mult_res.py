#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:29:52 2023

@author: bemi


Down block



"""




from flax import linen as nn
import functools

from jaxtyping import Array

from src.nn_models.nn_blocks import ResnetBlock



class DownBlock(nn.Module):
    """
    A helper Module with num_res_blocks ResNet blocks and optional 1 MaxPool or 1 down convolution.
    """
    block_name: str
    out_channels: int
    use_down_conv: bool
    pooling: bool = True
    p_dropout: float = 0.1
    num_res_blocks: int = 1
    
    def setup(self):        
        self.res_blocks = [ResnetBlock(
            p_dropout=self.p_dropout, 
            out_ch=self.out_channels, 
            name=f'{self.block_name}_resnet_block{i}')
            for i in range(self.num_res_blocks)]
        
        if self.pooling and self.use_down_conv:
            raise NotImplementedError('Currently system expects block to use either maxpool or downconv not both')
        
        if self.pooling:
            self.max_pool = functools.partial(
                nn.max_pool, 
                window_shape = (2, 2), 
                strides = (2, 2))
            
        if self.use_down_conv:
            self.down_conv = nn.Conv(
                features= self.out_channels, 
                kernel_size=(2, 2), 
                strides=(2, 2))            
            
    
    @nn.compact
    def __call__(self, x: Array, cond: Array, deterministic: bool):
        
        h = x
        for res in self.res_blocks:
            h = res(h, cond = cond, deterministic = deterministic)[0]
                
        before_pooling = h
        
        if self.pooling:
            h = self.max_pool(h)
        
        if self.use_down_conv:
            h = self.down_conv(h)
                
        return h, before_pooling
          
        
        




