#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:34:51 2023

@author: bemi


The UNet part of the ScoreUNet from:
    https://github.com/google-research/vdm/blob/main/model_vdm.py



"""


from jax import numpy as jnp

from flax import linen as nn
from jaxtyping import Array

from src.config_classes.nn_config import NNConfig
from src.nn_models.nn_blocks import AttnBlock, ResnetBlock



class VDMUNet(nn.Module):
    config: NNConfig
    
        
    @nn.compact    
    def __call__(self, h: Array, cond: Array, deterministic=True):
        config = self.config
        n_embd = config.n_embd

        hs = [h]
          
        # Downsampling
        for i_block in range(self.config.n_layer):
          block = ResnetBlock(p_dropout=config.p_dropout, out_ch=n_embd, name=f'down.block_{i_block}')
          h = block(hs[-1], cond, deterministic)[0]
          if config.with_attention:
              h = AttnBlock(num_heads=1, name=f'down.attn_{i_block}')(h)
          hs.append(h)
          
        # Middle
        h = hs[-1]
        h = ResnetBlock(config.p_dropout, name='mid.block_1')(h, cond, deterministic)[0]
        h = AttnBlock(num_heads=1, name='mid.attn_1')(h)
        h = ResnetBlock(config.p_dropout, name='mid.block_2')(h, cond, deterministic)[0]
          
        # Upsampling
        for i_block in range(self.config.n_layer + 1):
          b = ResnetBlock(p_dropout=config.p_dropout, out_ch=n_embd, name=f'up.block_{i_block}')
          h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
          if config.with_attention:
              h = AttnBlock(num_heads=1, name=f'up.attn_{i_block}')(h)
          
        assert not hs

        
        return h
            
         
