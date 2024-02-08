#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:57:05 2022

@author: bemi


Util functions for encoders/decoders


"""


import jax.numpy as jnp
from numpy.typing import NDArray

from typing import Tuple



def transform_values(x: NDArray, norm_const: float):
    return 2 * ((x + 0.5)/ norm_const) - 1


def create_checkerboard_mult_mask(h: int, w: int, channels: int):    
    x, y = jnp.arange(h, dtype=jnp.int32), jnp.arange(w, dtype=jnp.int32)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    mask = jnp.fmod(xx + yy, 2)
    mask = mask.astype(jnp.float32).reshape(1, h, w, 1)
    mask = jnp.tile(mask, (1, 1, 1, channels))
    return mask



def checkerboard(shape: Tuple):
    return jnp.indices(shape).sum(axis=0) % 2


def get_height_width_checkerboard_mask(height: int, width: int):
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f'Only even height and width supported. Given height: {height}, width: {width}')
    
    single_channel_mask = checkerboard((height, width))
    
    return single_channel_mask
    




