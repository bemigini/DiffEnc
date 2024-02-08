#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:54:44 2023

@author: bemi


Plotting diffusion loss per t


"""


import matplotlib.pyplot as plt
import numpy as np

import os
import re
from typing import Dict, List

from src.file_handling import save_load_json as sljson




def plot_loss_folder(folder: str, plot_dir: str, pattern: str = ''):
    loss_files = [file 
                    for file in os.listdir(folder)
                    if re.match(f'.*{pattern}.*_evaluation_metrics.*.json', file)]
    
    plot_losses_per_t(folder, loss_files, plot_dir, pattern)


def plot_losses_per_t(load_from_folder: str, loss_files: List[str], 
                      plot_dir: str, pattern: str):
    xy = np.array([])
    model_names = []
    
    green_tuple = ('#006d2c', '#31a354', '#74c476', '#a1d99b')
    blue_tuple = ('#08519c', '#3182bd', '#6baed6', '#9ecae1')
    
    for i, file in enumerate(loss_files): 
        model_name = file.split('.')[0]
        model_names.append(model_name)
        
        load_from_path = os.path.join(load_from_folder, file)
        json_metrics = sljson.load_json(load_from_path)
                
        current_xy = per_t_loss_dict_to_array(json_metrics, 'bpd_diff')
        if np.size(xy) == 0:
            xy = np.zeros((current_xy.shape[0], len(loss_files) + 1))
            xy[:, 0:2] = current_xy
        else:
            x = xy[:, 0]
            current_x = current_xy[:, 0]
            if x.shape[0] != current_x.shape[0] or (x != current_x).any():
                raise ValueError(f'Different x-values found: {xy[:, 0]} \n {current_xy[:, 0]}')
            
            xy[:, i+1] = current_xy[:, 1]            
    
    for i in range(1, xy.shape[1]):
        label = 'vdm' if 'VDM2' in model_names[i-1] else 'edm'
        color = blue_tuple[1] if label == 'vdm' else green_tuple[1]
        plt.plot(xy[:, 0], xy[:, i], color = color, label=label)
          
        
    plt.grid()  
    plt.legend()
    
    if plot_dir == '':
        plt.show()
    else:
        steps = '_'.join([k for k in json_metrics])
        plot_file_name = f'{pattern}_{steps}_diff_loss_per_t.png'
        sample_path = os.path.join(plot_dir, plot_file_name)
        plt.savefig(sample_path, dpi = 300)
        plt.clf()
    
    if 'cifar' in pattern:
        for i in range(1, xy.shape[1]):
            label = 'vdm' if 'VDM2' in model_names[i-1] else 'edm'
            color = blue_tuple[1] if label == 'vdm' else green_tuple[1]
            plt.plot(xy[-4:, 0], xy[-4:, i], color = color, label=label)
              
            
        plt.grid()  
        plt.legend()
        
        if plot_dir == '':
            plt.show()
        else:
            steps = '_'.join([k for k in json_metrics])
            plot_file_name = f'{pattern}_{steps}_late_t_diff_loss_per_t.png'
            sample_path = os.path.join(plot_dir, plot_file_name)
            plt.savefig(sample_path, dpi = 300)
            plt.clf()
        
    



def per_t_loss_dict_to_array(json_dict_metrics: Dict, loss_type: str):
    losses_step_dict = next(iter(json_dict_metrics.values()))
    
    losses = {
        float(k): losses_step_dict[k]['eval'][loss_type] 
        for k in losses_step_dict}

    losses_sorted = sorted(losses.items()) # sorted by key, return a list of tuples
    loss_xy = np.array(losses_sorted)
    
    return loss_xy

