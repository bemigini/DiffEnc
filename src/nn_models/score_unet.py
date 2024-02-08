#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:14:39 2022

@author: bemi


Score U-Net 


From: https://github.com/google-research/vdm 


# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""


from flax import linen as nn

from jax import numpy as jnp
from jaxtyping import Array


from src.nn_models.nn_blocks import AttnBlock, ResnetBlock
from src.nn_models.fourierfeatures import Base2FourierFeatures
from src.nn_models.time_embedding import get_timestep_embedding


from src.config_classes.nn_config import NNConfig
from src.nn_models.time_embedding import TimeEmbedding
from src.nn_models.time_embedding_t import TimeEmbeddingT
from src.nn_models.unet import UNet 
from src.nn_models.kingma_unet import VDMUNet




class ScoreUNet(nn.Module):
    config: NNConfig
    gamma_min: float
    gamma_max: float
    
    @nn.compact
    def __call__(self, 
                 z: Array, g_t: Array, conditioning: Array, 
                 deterministic=True):
        config = self.config
        
        # Compute conditioning vector based on 'g_t' and 'conditioning'
        n_embd = config.n_embd
          
        TE = TimeEmbedding(n_embd, self.gamma_min, self.gamma_max)        
        cond = TE(z, g_t, conditioning = conditioning)
                  
        # Concatenate Fourier features to input
        if config.with_fourier_features:
            z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
            h = jnp.concatenate([z, z_f], axis=-1)
            
        else:
            h = z
          
        # Linear projection of input
        h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
        
        vdm_unet = VDMUNet(config)   
        h = vdm_unet(h, cond, deterministic)
          
        # Predict noise
        normalize = nn.normalization.GroupNorm(self.config.num_groups_groupnorm)
        h = nn.swish(normalize(h))
        pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)
          
        # Base measure
        pred += z
          
        return pred


class ScoreUNetAlt(nn.Module):
    config: NNConfig
    gamma_min: float
    gamma_max: float
    
    @nn.compact
    def __call__(self, 
                 z: Array, g_t: Array, conditioning: Array, 
                 deterministic=True):
        config = self.config
        
        # Compute conditioning vector based on 'g_t' and 'conditioning'
        n_embd = self.config.n_embd
        
        TE = TimeEmbedding(n_embd, self.gamma_min, self.gamma_max)        
        cond = TE(z, g_t, conditioning = conditioning)
        
          
        # Concatenate Fourier features to input
        if config.with_fourier_features:
            z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
            h = jnp.concatenate([z, z_f], axis=-1)
            
        else:
            h = z
        
        
        h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
                        
        unet = UNet(config)        
        h = unet(h, cond, deterministic)
                
        # Predict noise
        normalize = nn.normalization.GroupNorm(self.config.num_groups_groupnorm)
        h = nn.swish(normalize(h))
        pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)
          
        # Base measure
        pred += z
          
        return pred


class ScoreUNetT(nn.Module):
    config: NNConfig
    
    @nn.compact
    def __call__(self, 
                 z: Array, t: Array, conditioning: Array, 
                 deterministic=True):
        config = self.config
        
        # Compute conditioning vector based on 't' and 'conditioning'
        n_embd = config.n_embd
                  
        assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        if jnp.isscalar(t) or (len(t.shape) == 1 and t.shape[0] == 1):
            t = jnp.ones((z.shape[0],), z.dtype) * t
        elif len(t.shape) == 0:
            t = jnp.tile(t[None], z.shape[0])
          
        temb = get_timestep_embedding(t, n_embd)
        cond = jnp.concatenate([temb, conditioning[:, None]], axis=1)
        cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
        cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))
          
        # Concatenate Fourier features to input
        if config.with_fourier_features:
            z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
            h = jnp.concatenate([z, z_f], axis=-1)
            
        else:
            h = z
          
        # Linear projection of input
        h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
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
          
        # Predict noise
        normalize = nn.normalization.GroupNorm(self.config.num_groups_groupnorm)
        h = nn.swish(normalize(h))
        pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)
          
        # Base measure
        pred += z
          
        return pred


class ScoreUNetAltT(nn.Module):
    config: NNConfig
    
    @nn.compact
    def __call__(self, 
                 z: Array, t: Array, conditioning: Array, 
                 deterministic=True):
        config = self.config
        
        # Compute conditioning vector based on 'g_t' and 'conditioning'
        n_embd = self.config.n_embd
        
        TE = TimeEmbeddingT(n_embd)        
        cond = TE(z, t, conditioning = conditioning)
          
        # Concatenate Fourier features to input
        if config.with_fourier_features:
            z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
            h = jnp.concatenate([z, z_f], axis=-1)
            
        else:
            h = z
        
        
        h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
                        
        unet = UNet(config)
        
        h = unet(h, cond, deterministic)
                
        # Predict noise
        normalize = nn.normalization.GroupNorm(self.config.num_groups_groupnorm)
        h = nn.swish(normalize(h))
        pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)
          
        # Base measure
        pred += z
          
        return pred







