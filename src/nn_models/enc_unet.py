#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:29:52 2023

@author: bemi


U-net for the trained encoder


"""




from flax import linen as nn


from typing import Dict

from src.nn_models.downblock import DownBlock
from src.nn_models.nn_blocks import AttnBlock, ResnetBlock
from src.nn_models.upblock import UpBlock


class UNet(nn.Module):
    config: Dict # TODO: Make NN config
    
    def setup(self):
        
        self.down_convs = []
        self.up_convs = [] 
        config = self.config
        
        # Down blocks
        for i in range(config.n_layer):
            out_channels = self.start_filters * (2 ** i)
            down_block = DownBlock(
                f'down_{i}', out_channels, config.pooling, config.p_dropout)
            self.down_convs.append(down_block)
        
        # Middle blocks
        self.mid_res1 = ResnetBlock(config.p_dropout, name='mid_resnet_1')
        self.mid_attn = AttnBlock(num_heads=1, name='mid_attn_1')
        self.mid_res2 = ResnetBlock(config.p_dropout, name='mid_resnet_2')
        
        # Up blocks
        for i in reversed(range(config.n_layer)):
            features = self.start_filters * (2 ** i)
            up_block = UpBlock(
                f'up_{i}', features, config.p_dropout)
            self.up_convs.append(up_block)
        
        self.final_res = ResnetBlock(config.p_dropout, name='final_resnet')
        
        
    




