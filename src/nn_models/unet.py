#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:29:52 2023

@author: bemi


U-net with max pooling and resnet blocks 


"""




from flax import linen as nn
from jaxtyping import Array

from src.config_classes.nn_config import NNConfig
from src.nn_models.downblock import DownBlock
from src.nn_models.nn_blocks import AttnBlock, ResnetBlock
from src.nn_models.upblock import UpBlock


class UNet(nn.Module):
    config: NNConfig
    
    def setup(self):
        
        config = self.config
        
        if config.channel_scaling == 'double':
            out_channels = lambda i: config.n_embd * (2 ** i)
        else:
            out_channels = lambda i: config.n_embd
        
        
        self.down_blocks = [
            DownBlock(
                f'down_{i}', out_channels(i), 
                config.down_conv,
                config.pooling, 
                config.p_dropout)
            for i in range(config.n_layer)]
        
        
        # Middle blocks
        self.mid_res1 = ResnetBlock(config.p_dropout, 
                                    num_groups_groupNorm = config.num_groups_groupnorm, 
                                    name='mid_resnet_1')
        self.mid_attn = AttnBlock(num_heads=1, name='mid_attn_1')
        self.mid_res2 = ResnetBlock(config.p_dropout,
                                    num_groups_groupNorm = config.num_groups_groupnorm, 
                                    name='mid_resnet_2')
        
        
        self.up_blocks = [
            UpBlock(
                f'up_{i}', out_channels(i), 
                config.down_conv or config.pooling, 
                config.p_dropout)
            for i in reversed(range(config.n_layer))
            ]
        
        
        self.final_res = ResnetBlock(config.p_dropout,
                                     num_groups_groupNorm = config.num_groups_groupnorm, 
                                     name='final_resnet')
        
        
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
            
            
            
            
        
        
        
    




