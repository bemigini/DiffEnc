#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:20:29 2023

@author: bemi

save and load hdf5

"""



import h5py

from numpy.typing import NDArray



def save_to_hdf5(data: NDArray, h5_file_path: str, h5_dataset_name: str) -> None:
    with h5py.File(h5_file_path, 'w') as h5f:
        h5f.create_dataset(h5_dataset_name, data=data, compression='gzip')
    

def load_from_hdf5(h5_file_path: str, dataset_name: str) -> NDArray:
    with h5py.File(h5_file_path, 'r') as h5f:
        data = h5f[dataset_name][:]
    
    return data