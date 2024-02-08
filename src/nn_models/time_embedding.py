#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:40:21 2023

@author: bemi


Time embedding based on gamma_t


"""



from flax import linen as nn
import jax
from jax import numpy as jnp
from jaxtyping import Array


class TimeEmbedding(nn.Module):
    n_embd: int 
    gamma_min: float
    gamma_max: float
    
    @nn.compact
    def __call__(self, z: Array, g_t: Array, conditioning: Array):        
        # Compute conditioning vector based on 'g_t' and 'conditioning'
        n_embd = self.n_embd
         
        lb = self.gamma_min
        ub = self.gamma_max
        t = (g_t - lb) / (ub - lb)  # ---> [0,1]
          
        assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        if jnp.isscalar(t) or (len(t.shape) == 1 and t.shape[0] == 1):
            t = jnp.ones((z.shape[0],)) * t
        elif len(t.shape) == 0:
            t = jnp.tile(t[None], z.shape[0])
          
        temb = get_timestep_embedding(t, n_embd)
                
        t_cond_emb = jnp.concatenate([temb, conditioning[:, None]], axis=1)
        t_cond_emb = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(t_cond_emb))
        t_cond_emb = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(t_cond_emb))
        
        return t_cond_emb


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
    """Build sinusoidal embeddings (from Fairseq).
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    Args:
      timesteps: jnp.ndarray: generate embedding vectors at these timesteps
      embedding_dim: int: dimension of the embeddings to generate
      dtype: data type of the generated embeddings
    Returns:
      embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(timesteps.shape) == 1
    timesteps *= 1000.
    
    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
      emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb




