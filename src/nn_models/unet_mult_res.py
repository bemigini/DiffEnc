#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:29:52 2023

@author: bemi


U-net with max pooling and resnet blocks 


"""




import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array
import math

from src.config_classes.nn_config import NNConfig
from src.nn_models.downblock_mult_res import DownBlock
from src.nn_models.nn_blocks import AttnBlock, ResnetBlock
from src.nn_models.upblock_mult_res import UpBlock


class UNetMultRes(nn.Module):
    config: NNConfig
    input_height: int
    
    def setup(self):
        
        config = self.config
        down_res = config.n_layer
        down_layers = config.n_layer
        
        for i in range(down_res):
            new_height = self.input_height / 2 **(i+1)
            if new_height % 2 == 1 or new_height <= 8:
                down_layers = i + 1
                break
        
        if down_layers != down_res:
            res_per_layer = int(math.floor(down_res / down_layers))
        else:
            res_per_layer = 1
        
        
        if config.channel_scaling == 'double':
            out_channels = lambda i: config.n_embd * (2 ** i)
        else:
            out_channels = lambda i: config.n_embd
        
        
        self.down_blocks = [
            DownBlock(
                f'down_{i}', out_channels(i), 
                config.down_conv,
                config.pooling, 
                config.p_dropout,
                res_per_layer)
            for i in range(down_layers)]
        
        
        # Middle blocks
        self.mid_res1 = ResnetBlock(config.p_dropout, name='mid_resnet_1')
        self.mid_attn = AttnBlock(num_heads=1, name='mid_attn_1')
        self.mid_res2 = ResnetBlock(config.p_dropout, name='mid_resnet_2')
        
        
        self.up_blocks = [
            UpBlock(
                f'up_{i}', out_channels(i), 
                config.down_conv or config.pooling, 
                config.p_dropout,
                res_per_layer)
            for i in reversed(range(down_layers))
            ]
        
        
        self.final_res = ResnetBlock(config.p_dropout, name='final_resnet')
        
        
    @nn.compact    
    def __call__(self, x: Array, cond: Array, deterministic: bool):
        outputs = []
        
        for module in self.down_blocks:
            x, before_pooling = module(x, cond, deterministic)
            outputs.append(before_pooling)
                
        x = self.mid_res1(x, cond = cond, deterministic = deterministic)[0]
        x = self.mid_attn(x)
        x = self.mid_res2(x, cond = cond, deterministic = deterministic)[0]
        
        for module in self.up_blocks:
            before_pool = outputs.pop()
            x = module(x, before_pool, cond, deterministic)
        
        # Kingma does not have this block
        # x = self.final_res(x, cond = cond, deterministic = deterministic)[0]
        
        return x
            
            
            
            
        
        
        
    




