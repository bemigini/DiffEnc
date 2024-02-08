#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 08:51:52 2023

From: https://github.com/google-research/vdm/blob/main/utils.py
"""
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


import functools 

import flax
import jax 



def dist(fn, accumulate: str, axis_name='devices'):
    """Wrap a function in pmap and device_get(unreplicate(.)) its return value."""
    
    if accumulate == 'concat':
      accumulate_fn = functools.partial(
          allgather_and_reshape, axis_name=axis_name)
    elif accumulate == 'mean':
      accumulate_fn = functools.partial(
          jax.lax.pmean, axis_name=axis_name)
    elif accumulate == 'none':
      accumulate_fn = None
    else:
      raise NotImplementedError(accumulate)
    
    @functools.partial(jax.pmap, axis_name=axis_name)
    def pmapped_fn(*args, **kwargs):
      out = fn(*args, **kwargs)
      return out if accumulate_fn is None else jax.tree_map(accumulate_fn, out)
    
    def wrapper(*args, **kwargs):
      return jax.device_get(
          flax.jax_utils.unreplicate(pmapped_fn(*args, **kwargs)))
    
    return wrapper


def allgather_and_reshape(x, axis_name='devices'):
    """Allgather and merge the newly inserted axis w/ the original batch axis."""
    y = jax.lax.all_gather(x, axis_name=axis_name)
    assert y.shape[1:] == x.shape
    return y.reshape(y.shape[0] * x.shape[0], *x.shape[1:])