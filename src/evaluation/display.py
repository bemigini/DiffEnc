#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:14:25 2023

@author: bemi


Utilities for displaying images


"""


import os

import jax.numpy as jnp
from jaxtyping import Array

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from typing import List, Tuple

from src.file_handling import naming, save_load_hdf5


def make_image_grid(images, 
                    nrows: int, 
                    ncols: int, 
                    model_name: str, 
                    col_labels: List[str] = [],
                    row_labels: List[str] = [],
                    cmap: str = '',
                    axes_pad: float = 0.1,
                    fontsize: str = 'medium',
                    text_x_y: Tuple[float, float] = (2.5, -10.0)):
    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, 
                     nrows_ncols=(nrows, ncols),  
                     axes_pad=axes_pad,  
                     label_mode = 'L')
    
    if col_labels:
        for i in range(ncols):
            ax = grid[i]
            ax.text(text_x_y[0], text_x_y[1], col_labels[i], 
                    fontsize=fontsize, 
                    ha = 'left',
                    verticalalignment='top')
    if row_labels:
        for i in range(nrows):
            ax = grid[i*ncols]
            ax.text(-70., 17., row_labels[i], 
                    fontsize=fontsize, 
                    ha = 'left',
                    verticalalignment='center')
    
    for ax, im in zip(grid, images):
        if '_mnist_' in  model_name or '_lego_' in model_name:
            ax.imshow(im, cmap='gray')
        elif cmap != '':
            ax.imshow(im, cmap = cmap)
        else:
            ax.imshow(im)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    

    return fig


def make_image_grid_special_first_col(images, 
                    nrows: int, 
                    ncols: int, 
                    model_name: str, 
                    col_labels: List[str] = [],
                    cmap: str = '',
                    fontsize: str = 'medium'):
    fig = plt.figure(figsize=(7, 7))
    grid = ImageGrid(fig, 111, 
                     nrows_ncols=(nrows, ncols),  
                     axes_pad=0.05,  
                     )
    
    for i, (ax, im) in enumerate(zip(grid, images)):
        if i % ncols == 0:
            ax.imshow(im)
        else:
            ax.imshow(im, cmap = cmap)
        
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    
    if col_labels:
        for i in range(ncols):
            ax = grid[i]
            ax.text(15.0, -10.0, col_labels[i], 
                    fontsize = fontsize, 
                    ha = 'center',
                    verticalalignment='top')

    return fig



def move_samples_to_m1to1(samples: Array, scale_up_small: bool):
    samples_mins = jnp.min(samples, axis = [1,2,3])
    move_to_zero_diff = jnp.minimum(samples_mins, 0).reshape(*samples_mins.shape, 1, 1, 1)

    moved_samples = samples + jnp.abs(move_to_zero_diff)
    
    moved_max = jnp.max(moved_samples, axis = [1,2,3]).reshape(*samples_mins.shape, 1, 1, 1)
    rescaled_samples = jnp.where(jnp.logical_or(scale_up_small, moved_max > 1), 
                                 moved_samples/moved_max, 
                                 moved_samples)
    
    return rescaled_samples


def move_samples_to_0to1_global(samples: Array, scale_up_small: bool):
    samples_min = jnp.min(samples)
    move_to_zero_diff = jnp.minimum(samples_min, 0)

    moved_samples = samples + jnp.abs(move_to_zero_diff)
    
    if scale_up_small:
        moved_max = jnp.max(moved_samples)
    else:
        moved_max = max(jnp.max(moved_samples), 1)
    rescaled_samples = moved_samples/moved_max
    
    return rescaled_samples


def move_samples_to_mean_05_global(samples: Array, scale_up_small: bool):
    samples_mean = jnp.mean(samples)
    
    moved_samples = (samples - samples_mean)
    
    if scale_up_small:
        moved_max = jnp.max(jnp.abs(moved_samples))
    else:
        moved_max = max(jnp.max(jnp.abs(moved_samples)), 0.5)
    
    rescaled_samples = moved_samples/(2*moved_max)
    
    mean_05_samples = rescaled_samples + 0.5
    
    return mean_05_samples



# h_5_path = os.path.join(main_folder, sample_folder, sample_file)
# 
# dataset = 'cifar10'
# dataset = '_mnist_'
#
# sample_file = '2023-08-22_S3VA_fixed_-13_3_5_0_unet_8_unet_2_m1_to_1_cifar10_2_128_1400000_samples_256_000.h5'
# sample_file = '2023-08-22_S3VA_fixed_-13_3_5_0_unet_8_unet_2_m1_to_1_cifar10_2_128_2000000_samples_256_000.h5'
# sample_file = '2023-09-25_S3VA_fixed_-13_3_5_0_unet_32_unet_mult_res_4_m1_to_1_cifar10_2_128_5000000_samples_256_000.h5'
# sample_file = '2023-09-25_S3VA_fixed_-13_3_5_0_unet_32_unet_mult_res_4_m1_to_1_cifar10_2_128_8000000_samples_256_000.h5'
# sample_file = '2023-09-25_VDM2_V_fixed_-13_3_5_0_unet_32_simple_cifar10_2_128_5000000_samples_256_000.h5'
# sample_file = '2023-09-25_VDM2_V_fixed_-13_3_5_0_unet_32_simple_cifar10_2_128_8000000_samples_256_000.h5'
def display_samples_from_h5(h_5_path: str, dataset: str, save_dir: str):
    h5_dataset_name = naming.get_samples_h5_dataset_name()
    loaded_samples = save_load_hdf5.load_from_hdf5(h_5_path, h5_dataset_name)
    
    loaded_shape = loaded_samples.shape 
    
    for i in range(loaded_shape[0]):
        current_samples = loaded_samples[i]
        if current_samples.max() > 50:
            current_samples = current_samples.astype(int)
        
        # TODO: let number of displayed samples and dataset be chosen
        rows = 5
        cols = 20
        make_image_grid(current_samples[0:100], rows, cols, dataset, axes_pad = 0.0)
        
        model_name = h_5_path.split('/')[-1].split('samples')[0]
        
        im_file_name = f'{model_name}{rows}_{cols}_tn_{i}.png'
        im_path = os.path.join(save_dir, im_file_name)
        if not os.path.exists(im_path):
            plt.savefig(im_path, dpi = 300, bbox_inches='tight')
                
        plt.show()
       

# sample_files = [
#   '2023-08-22_S3VA_fixed_-13_3_5_0_unet_8_unet_2_m1_to_1_cifar10_2_128_1400000_samples_256_000.h5',
#   '2023-09-25_S3VA_fixed_-13_3_5_0_unet_32_unet_mult_res_4_m1_to_1_cifar10_2_128_5000000_samples_256_000.h5',
#   '2023-09-25_VDM2_V_fixed_-13_3_5_0_unet_32_simple_cifar10_2_128_5000000_samples_256_000.h5'
#]
# sample_files = [
#   '2023-08-22_S3VA_fixed_-13_3_5_0_unet_8_unet_2_m1_to_1_cifar10_2_128_1400000_samples_256_000.h5',
#   '2023-09-25_S3VA_fixed_-13_3_5_0_unet_32_unet_mult_res_4_m1_to_1_cifar10_2_128_8000000_samples_256_000.h5',
#   '2023-09-25_VDM2_V_fixed_-13_3_5_0_unet_32_simple_cifar10_2_128_8000000_samples_256_000.h5'
#]
def display_comparison_samples_from_h5(
        sample_folder: str, 
        sample_files: List[str], 
        dataset: str, 
        save_dir: str,
        row_labels: List[str] = ['DiffEnc-8-2', 'DiffEnc-32-4', 'VDMv-32'],
        fontsize: str = 'x-small'):
    
    h_5_paths = [os.path.join(sample_folder, sample_file)
                 for sample_file in sample_files]
    h5_dataset_name = naming.get_samples_h5_dataset_name()
    
    sample_rows = len(h_5_paths)
    same_model_rows = 1
    cols = 13
    num_same_model_samples = same_model_rows * cols
    sample_array = np.array([])
    
    
    for j, h_5_path in enumerate(h_5_paths):
        loaded_samples = save_load_hdf5.load_from_hdf5(h_5_path, h5_dataset_name)
    
        loaded_shape = loaded_samples.shape
        i = loaded_shape[0] - 1
        if len(sample_array) == 0:
            sample_array = np.zeros(
                (sample_rows, num_same_model_samples, *loaded_shape[2:]),
                dtype = int)
        
        current_samples = loaded_samples[i].astype(int)
        sample_array[j, :] = current_samples[0:num_same_model_samples]
        
    
    stacked_samples = np.vstack(sample_array)
    
    make_image_grid(stacked_samples, sample_rows, cols, dataset, 
                    row_labels = row_labels,
                    axes_pad = (0.0, 0.05),
                    fontsize = fontsize)
    
    comparison_name = 'Comp_' + '_'.join(row_labels)
    
    im_file_name = f'{comparison_name}_{sample_rows}_{cols}.png'
    im_path = os.path.join(save_dir, im_file_name)
    if not os.path.exists(im_path):
        plt.savefig(im_path, dpi = 300, bbox_inches='tight')
        
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    


