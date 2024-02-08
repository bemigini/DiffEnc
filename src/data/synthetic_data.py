#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:36:26 2022

@author: bemi


Synthetic data 


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



#import jax.numpy as jnp
import numpy as np
#import torch.utils.data as data


def make_8_bit_swirl_data(size: int, seed: int):
    np_rng = np.random.RandomState(seed=seed)
    # Make 8-bit swirl dataset
    theta = np.sqrt(np_rng.random(size))*3*np.pi # np.linspace(0,2*pi,100)
    r_a = 2*theta + np.pi
    x = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    # We use 8 bits, to make this a bit similar to image data, which has 8-bit
    # color channels.
    x = 4*(x + .25*np_rng.normal(size,2) + 30)
    x = x.astype('uint8')
    
    return x 


"""
class SwirlDataset(data.Dataset):

    def __init__(self, size: int, seed:int):
        
        #Inputs:
        #    size - Number of data points we want to generate
        #    seed - The seed to use to create the PRNG state with which we want to generate the data points
        
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed=seed)
        self.data = self.make_8_bit_swirl_data()
        
    # N: Number of datapoints
    def make_8_bit_swirl_data(self):
        # Make 8-bit swirl dataset
        theta = np.sqrt(self.np_rng.random(self.size))*3*np.pi # np.linspace(0,2*pi,100)
        r_a = 2*theta + np.pi
        x = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        # We use 8 bits, to make this a bit similar to image data, which has 8-bit
        # color channels.
        x = 4*(x + .25*self.np_rng.normal(self.size,2) + 30)
        x = x.astype('uint8')
        
        return x 
    

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = jnp.zeros((1,))
        return data_point, data_label



"""







