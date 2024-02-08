#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:28:53 2023

@author: bemi


Sample dataset

Dataset class which makes generated samples into a tensoflow dataset.


"""


import numpy as np
import os
import re

from src.file_handling import naming, save_load_hdf5



class SampleDatasetGenerator:
    def __init__(self, file_prefix: str, folder_path: str):
        files = [file
                 for file in os.listdir(folder_path)
                 if re.match(f'{file_prefix}.*.h5', file)]
        
        if len(files) == 0:
            raise ValueError(f'No sample files found with prefix: {file_prefix}')
        
        self.folder = folder_path
        self.files = files         
    
    
    def __iter__(self):
      self.current_file_idx = 0
      return self
                 
    
    def __next__(self):
        if self.current_file_idx < len(self.files):   
            file = self.files[self.current_file_idx]
            
            h5_dataset_name = naming.get_samples_h5_dataset_name()
            h_5_path = os.path.join(self.folder, file)
            loaded_samples = save_load_hdf5.load_from_hdf5(h_5_path, h5_dataset_name)
            
            # We expect the finished samples to be at index 11
            idx = 11
            images = loaded_samples[idx]
            images = images.astype(int)
            
            unconditional = np.zeros(images.shape[0])
            
            self.current_file_idx += 1
            return images, unconditional
        else:
          raise StopIteration
        
















