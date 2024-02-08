#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:49:46 2023

@author: bemi


Base class for encoder and decoder of z_0

Based on https://github.com/google-research/vdm/blob/main/model_vdm.py



"""




from flax import linen as nn
import jax
from jax import numpy as jnp


from jaxtyping import Array
from jax._src.random import PRNGKey


from numpy.typing import NDArray



from src.config_classes.enc_dec_config import EncDecConfig
from src.encode_decode import util as enc_dec_util 


class EncDecBase(nn.Module):
    config: EncDecConfig
    
    # x: The data. First dimension is samples. 
    #    Images are expected to be (height x width x channels).
    # g_0: Noise schedule at time zero, array only one element 
    def __call__(self, x: NDArray, g_0: Array):
        encoded = self.initial_encode(x)
        return self.decode_to_logprobs(encoded, g_0)
    
    def initial_encode(self, x: NDArray):
        # This transforms values in x to the domain (-1,1).
        # Rounding here just a safeguard to ensure the input is discrete
        # (although typically, x is a discrete variable such as uint8)
        x = x.round() 
        # We expect values in x to go from 0 to self.config.vocab_size - 1
        # in discrete values.
                
        return enc_dec_util.transform_values(x, self.config.vocab_size)
    
    
    # z_0_rescaled: Encoded data at time 0
    # g_0: Noise schedule at time zero, array with only one element      
    def decode_to_logprobs(self, z_0_rescaled: Array, gamma_0: Array):
        channels = z_0_rescaled.shape[-1]
        # Logits are exact if there are no dependencies between dimensions of x
        x_vals = jnp.arange(0, self.config.vocab_size).reshape(-1, 1)
        x_vals = jnp.repeat(x_vals, channels, 1)
        x_vals = self.initial_encode(x_vals)
        
        x_vals = x_vals.transpose([1, 0]).reshape(1, 1, 1, channels, -1)
        inv_stdev = jnp.exp(-0.5 * gamma_0)
        # Get density using log(normal PDF)
        logits = -0.5 * jnp.square((z_0_rescaled.reshape(*z_0_rescaled.shape, 1) - x_vals) * inv_stdev)
        
        # Using softmax to get the categorical distribution
        logprobs = jax.nn.log_softmax(logits)
        return logprobs
    
    
    # The decoder expects values in the interval (-1, 1), so z_0 must be 
    # rescaled accordingly
    def logprob(self, x: Array, z_0_rescaled: Array, gamma_0: Array):
        x = x.round().astype('int32')
        x_onehot = jax.nn.one_hot(x, self.config.vocab_size)
        logprobs = self.decode_to_logprobs(z_0_rescaled, gamma_0)
        logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3, 4))
        return logprob
    
    
    def generate_x(self, z_0: Array, gamma_0: Array, cat_rng: PRNGKey):
        var_0 = nn.sigmoid(gamma_0)
        z_0_rescaled = z_0 / jnp.sqrt(1. - var_0)
        logits = self.decode_to_logprobs(z_0_rescaled, gamma_0)
                
        if cat_rng is not None:
            samples = jax.random.categorical(cat_rng, logits)
        else:
            samples = jnp.argmax(logits, axis=-1)
        
        return samples
    
    
    
    
    
    
    
    
    
    
    
    

