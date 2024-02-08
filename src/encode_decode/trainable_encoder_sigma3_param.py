#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:12:16 2023

@author: bemi


An Encoder which can learn the encoding based on gamma_t.



"""


from flax import linen as nn

import jax
from jax import numpy as jnp
from jaxtyping import Array


from numpy.typing import NDArray


from src.config_classes.enc_dec_config import EncDecConfig
from src.encode_decode.encdec_base import EncDecBase
from src.nn_models.time_embedding_t import TimeEmbeddingT
from src.nn_models.unet import UNet 
from src.nn_models.kingma_unet import VDMUNet
from src.nn_models.unet_mult_res import UNetMultRes



class TrainableEncSigma3Param(EncDecBase):
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
                
        self.inner_encoder = InnerTrainableEncSigmaParam(self.config,
                                                         self.gamma_min,
                                                         self.gamma_max,
                                                         self.input_height)
        
    
    # x: The data. First dimension is samples. 
    #    Images are expected to be (height x width x channels).
    # g_t: Noise schedule at time t 
    @nn.compact
    def __call__(self, 
                 init_x: NDArray, 
                 gamma_t: Array, 
                 conditioning: Array, 
                 deterministic: bool):
        
        y_x_t = self.inner_encoder(init_x, gamma_t, conditioning, deterministic)        
        sigma_squared = nn.sigmoid(gamma_t.reshape(-1, 1, 1, 1))
        
        # f(x, t) = x + \sigma**2 g(x, t) - \sigma**2 x 
        x_g_t = init_x + sigma_squared * y_x_t - sigma_squared * init_x
            
        return x_g_t
    
    
    def get_inner_encoded(self, 
                 init_x: NDArray, 
                 t: Array, 
                 conditioning: Array, 
                 deterministic: bool):
        
        y_x_t = self.inner_encoder(init_x, t, conditioning, deterministic)
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


class InnerTrainableEncSigmaParam(EncDecBase):
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
        self.k = self.config.k
        self.use_k = self.k != 1. 
        self.init_var = self.config.init_var
        self.use_var_bound = self.init_var > 0
        self.min_var = self.config.min_var
        self.use_min_var = self.min_var > 0
        self.use_x_var = self.config.use_x_var
        
        self.num_groups_groupnorm = self.nn_config.num_groups_groupnorm
                
        if self.nn_config.m_type == 'unet':
            self.enc_model = UNet(self.nn_config)
        elif self.nn_config.m_type == 'vdm_unet':
            self.enc_model = VDMUNet(self.nn_config)
        elif self.nn_config.m_type == 'unet_mult_res':
            self.enc_model = UNetMultRes(self.nn_config, self.input_height)
        else:
            raise NotImplementedError(f'Model type not implemented: {self.nn_config.m_type}')
        
        self.time_embedding = TimeEmbeddingT(self.nn_config.n_embd)
    
    
    @nn.compact
    def __call__(self, 
                 init_x: NDArray, 
                 gamma_t: Array, 
                 conditioning: Array, 
                 deterministic: bool):
        
        lb = self.gamma_min
        ub = self.gamma_max
        t = (gamma_t - lb) / (ub - lb)  # ---> [0,1]
        
        t_emb = self.time_embedding(init_x, t, conditioning = conditioning)
        
        h = nn.Conv(
            features=self.nn_config.n_embd, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            name='conv_in')(init_x)
        
        h = self.enc_model(h, t_emb, deterministic)
        
        normalize = nn.normalization.GroupNorm(self.num_groups_groupnorm)
        h = nn.swish(normalize(h))
        h = nn.Conv(
            features=init_x.shape[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.zeros,
            name='conv_out')(h)
        
        
        if self.gamma_reg:
            # Mult with t in [0,1]            
            t = t.reshape(-1, 1, 1, 1)
            h = h * t
        if self.end_reg:
            # Mult with 1-t in [0,1]            
            t = t.reshape(-1, 1, 1, 1)
            h = h * (1.-t)
        if self.end_half_reg:
            # Mult with 0 for t > 0.5            
            t = t.reshape(-1, 1, 1, 1)
            h = h * jnp.max(0.5*t - t**2, 0)
        
        
        # Regulate how large the values of the inner encoded x can be
        if self.m1_to_1:
            h = jnp.tanh(h)
        
        if self.use_k:
            h = self.k * h
        
        if self.use_var_bound:            
            h_var = jnp.var(h, axis = [1, 2, 3])
            mult = jnp.sqrt(self.init_var / jnp.maximum(h_var, self.init_var)) 
            
            # New var of h will be the same as before where h_var <= var_bound
            # And otherwise new var will be var_bound
            h = h * jnp.reshape(mult, (-1, 1, 1, 1))
            if self.use_min_var:
                h_var = jnp.var(h, axis = [1, 2, 3])
                # Maximum inside to avoid division by zero
                mult = jnp.sqrt(self.min_var / jnp.minimum(jnp.maximum(h_var, 0.001), self.min_var)) 
                h = h * jnp.reshape(mult, (-1, 1, 1, 1))
        elif self.use_x_var:
            h_var = jnp.var(h, axis = [1, 2, 3])
            x_var = jnp.var(init_x, axis = [1, 2, 3])
            mult = jnp.sqrt(x_var / jnp.maximum(h_var, x_var))
            h = h * jnp.reshape(mult, (-1, 1, 1, 1))
                            
        
        # The inner encoder is the identity at step zero
        if self.id_at_zero:
            h = init_x + h
            if self.m1_to_1:
                x_max = jnp.max(jnp.abs(h), axis = [1, 2, 3])
                mult = 1/jnp.maximum(1, x_max)
                h = h * jnp.reshape(mult, (-1, 1, 1, 1))
        
        return h
    
    
    
    
    
    
    
    
    
    
    
    
    

