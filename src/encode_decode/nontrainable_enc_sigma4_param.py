#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:12:16 2023

@author: bemi


An Encoder based on gamma_t, but which is not trainable.



"""


from flax import linen as nn

import jax
from jax import numpy as jnp
from jaxtyping import Array


from numpy.typing import NDArray


from src.config_classes.enc_dec_config import EncDecConfig
from src.encode_decode.encdec_base import EncDecBase




class NonTrainableEncSigma4Param(EncDecBase):
    config: EncDecConfig
    gamma_min: float
    gamma_max: float
    input_height: int
    
    def setup(self):
        self.gamma_reg = self.config.gamma_reg
        self.end_reg = self.config.end_reg
        self.end_half_reg = self.config.end_half_reg
        
        self.id_at_zero = self.config.id_at_zero
        self.nn_config = self.config.nn_config
        self.m1_to_1 = self.config.m1_to_1
        self.num_groups_groupnorm = self.nn_config.num_groups_groupnorm
                
        
        
    
    # x: The data. First dimension is samples. 
    #    Images are expected to be (height x width x channels).
    # g_t: Noise schedule at time t 
    @nn.compact
    def __call__(self, 
                 init_x: NDArray, 
                 gamma_t: Array, 
                 conditioning: Array, 
                 deterministic: bool):
                
        sigma_squared = nn.sigmoid(gamma_t.reshape(-1, 1, 1, 1))
        
        # f(x, t) = x - \sigma_t**2 * x
        x_g_t = init_x - sigma_squared * init_x
            
        return x_g_t
    
    
    def get_inner_encoded(self, 
                 init_x: NDArray, 
                 t: Array, 
                 conditioning: Array, 
                 deterministic: bool):
        
        y_x_t = init_x
        return y_x_t    
            
    
    def x_t_to_rgb(self, x_t: Array):
        x_vals = jnp.arange(0, self.config.vocab_size).reshape(-1, 1)
        channels = x_t.shape[-1]
        x_vals = jnp.repeat(x_vals, channels, 1)
        x_vals = self.initial_encode(x_vals).transpose([1, 0]).reshape(1, 1, 1, channels, -1)
        
        diff = -jnp.abs(x_t.reshape(*x_t.shape, 1) - x_vals)
        
        logprobs = jax.nn.log_softmax(diff)
        
        samples = jnp.argmax(logprobs, axis=-1)
        
        return samples


    
    
    
    
    
    
    
    
    
    
    

