#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:06:03 2023

@author: bemi
"""


import os
import pickle
import math
import numpy as np

import tensorflow as tf


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo)
    return d


class ImageNetDatasetGenerator:
    def __init__(self, 
                 imagenet_folder: str, img_size: int, 
                 is_train: bool,
                 seed: int, train_steps: int, batch_size: int):
        
        self.imagenet_folder = imagenet_folder
        self.img_size = img_size
        self.is_train = is_train
        self.file_seed = seed
        self.within_file_seed = seed 
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.dataset_size = 1281167
        self.file_size = 128116
        
        if is_train:
            folder = os.path.join(imagenet_folder, f'Imagenet{img_size}_train')
        else:
            folder = os.path.join(imagenet_folder, f'Imagenet{img_size}_val')
        
        self.folder = folder
        
        files = [file for file in os.listdir(folder)]
        
        if len(files) == 0:
            raise ValueError(f'No files found in: {folder}')
        
        self.files = files
        self.file_indexes = np.arange(len(files))
        
        if self.is_train:
            steps_per_file = math.floor(self.file_size/batch_size)
            repeats = math.ceil(train_steps / (len(self.files)*steps_per_file))
            self.file_indexes = np.repeat(self.file_indexes, repeats)
            np.random.seed(self.file_seed)
            np.random.shuffle(self.file_indexes)
            
    
    def set_within_file_seed(self, seed):
        self.within_file_seed = seed
    
    
    def update_within_file_seed(self, seed):
        self.within_file_seed = self.file_seed + seed
    
    
    def __iter__(self):
        self.current_file_idx = 0
        return self
                 
    
    def __next__(self):
        if self.current_file_idx < len(self.file_indexes):   
            file = self.files[self.file_indexes[self.current_file_idx]]
            
            img_size2 = self.img_size * self.img_size
            
            data_file = os.path.join(self.folder, file)
            d = unpickle(data_file)
            x = d['data']
            
            x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
            images = x.reshape((x.shape[0], self.img_size, self.img_size, 3))
                        
            unconditional = np.zeros(images.shape[0])
            
            ds = tf.data.Dataset.from_tensor_slices((images, unconditional))
            
            if self.is_train:
                self.update_within_file_seed(self.current_file_idx)
                ds = ds.shuffle(
                    buffer_size = 2000, 
                    seed = self.within_file_seed, 
                    reshuffle_each_iteration=True)
            
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            
            self.current_file_idx += 1
            return ds
        else:
          raise StopIteration
        
