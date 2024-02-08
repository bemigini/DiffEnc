#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:52:09 2022

@author: bemi


Functions used for score models


Based on: https://github.com/google-research/vdm


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
import jax
from jax import numpy as jnp
from jaxtyping import Array

import numpy as np

from src.basic_ddpm.data_classes import VDMConfig
from src.nn_models.fourierfeatures import Base2FourierFeatures



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
    emb = np.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
      emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
    

def get_conditioning_vector(
        z: Array, 
        g_t: Array, 
        conditioning: Array, 
        config: VDMConfig):
    n_embd = config.sm_n_embd
    
    lb = config.gamma_min
    ub = config.gamma_max
    t = (g_t - lb) / (ub - lb)  # ---> [0,1]
  
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
    if jnp.isscalar(t):
        t = jnp.ones((z.shape[0],), z.dtype) * t
    elif len(t.shape) == 0:
        t = jnp.tile(t[None], z.shape[0])
  
    temb = get_timestep_embedding(t, n_embd)
    cond = jnp.concatenate([temb, conditioning[:, None]], axis=1)
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))
    
    return cond


def concatenate_gamma_t_as_feature():
    

    
def concatenate_fourier_features(z: Array, config: VDMConfig):
    if config.with_fourier_features:
        z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
        h = jnp.concatenate([z, z_f], axis=-1)
    else:
        h = z
    
    return h
    
    








