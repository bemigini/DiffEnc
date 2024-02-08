#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:28:09 2023

@author: bemi


Visualising training and test loss

Visualizing x_t grad.


"""


from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import os
import re
from scipy import stats
from typing import Dict, List, Tuple

from src.file_handling import save_load_json as sljson





def plot_losses(load_from_folder: str, load_from_file: str, plot_dir: str, zoom_min: List[int] = [200000], zoom_max: List[int] = [300000]):
    load_from_path = os.path.join(load_from_folder, load_from_file)
    json_metrics = sljson.load_json(load_from_path)
    model_name = load_from_file.split('.')[0]

    json_dict_metrics = {int(k):elm[k] for elm in json_metrics for k in elm}
    
    train_xy = loss_dict_to_array(json_dict_metrics, 'train', 'bpd_diff')
    test_xy = loss_dict_to_array(json_dict_metrics, 'eval', 'bpd_diff')
    
    plot_loss_with_zooms(train_xy, test_xy, loss_name = 'diff',
                         model_name = model_name, plot_dir = plot_dir,
                         zoom_min = zoom_min, zoom_max = zoom_max)
    
    train_re_xy = loss_dict_to_array(json_dict_metrics, 'train', 'bpd_recon')
    test_re_xy = loss_dict_to_array(json_dict_metrics, 'eval', 'bpd_recon')
    
    plot_loss_with_zooms(train_re_xy, test_re_xy, loss_name = 'recon',
                         model_name = model_name, plot_dir = plot_dir,
                         zoom_min = zoom_min, zoom_max = zoom_max)
        
        
    train_ltn_xy = loss_dict_to_array(json_dict_metrics, 'train', 'bpd_latent')
    test_ltn_xy =  loss_dict_to_array(json_dict_metrics, 'eval', 'bpd_latent')
    
    min_test_latent_loss_idx = np.argmin(test_ltn_xy[:, 1])
    
    test_min, test_max, test_min_step = get_min_max_y(
        test_ltn_xy,
        0, 500000)
    y_ltn_max = test_max
    y_ltn_min = test_min
    
    for i in range(3):
        if i == 1:
            if test_max < 0.00145:
                y_ltn_max = 0.00145
            
            if test_min > 0.00075:
                y_ltn_min = 0.00075
        if i == 2:
            min_use = 150000 if 150000 in test_ltn_xy[:, 0] else 0
            test_min, test_max, test_min_step = get_min_max_y(
                test_ltn_xy,
                min_use, 500000)
            if test_max < 0.06:
                y_ltn_max = 0.06
            
            if test_min > 0.0002:
                y_ltn_min = 0.0002
        plt.ylim([y_ltn_min - 0.0001, y_ltn_max + 0.0001])# TODO: change this 
        
        plt.plot(train_ltn_xy[:, 0], train_ltn_xy[:, 1], 'b', label = 'train')
        plt.plot(test_ltn_xy[:, 0], test_ltn_xy[:, 1], 'g', label = 'test')
        plt.legend()
        plt.grid()
        if 'trainable' in model_name:
            title = model_name.split('_unconditional')[0]
            title = f'{title.split("_LE")[1]}'
        else:
            title = model_name.split('_unconditional')[0]
            title = f'VDM{title.split("_VDM")[1]}'
        title = f'{title}_latent\n min test loss: {test_ltn_xy[:, 1][min_test_latent_loss_idx]}\n at step: {test_ltn_xy[:, 0][min_test_latent_loss_idx]}'
        plt.title(title, fontsize = 8)
        
        if plot_dir == '':
            plt.show()
        else:        
            plot_file_name = f'{model_name}_train_test_latent_loss_{i}.png'
            sample_path = os.path.join(plot_dir, plot_file_name)
            if not os.path.exists(sample_path):
                plt.savefig(sample_path, dpi = 300)
        
        plt.clf()
    
    
    # Plotting total loss     
    train_recon_losses = {k: np.array(
        json_dict_metrics[k]['train']['bpd_recon'])  
                    for k in json_dict_metrics}

    test_recon_losses = {k: np.array( 
         json_dict_metrics[k]['eval']['bpd_recon'])  
                    for k in json_dict_metrics if 'eval' in json_dict_metrics[k]}
    
    train_r_sorted = sorted(train_recon_losses.items()) # sorted by key, return a list of tuples
    train_r_xy = np.array(train_r_sorted)

    test_r_sorted = sorted(test_recon_losses.items()) # sorted by key, return a list of tuples
    test_r_xy = np.array(test_r_sorted)    
    
    total_train_loss = train_xy[:, 1] + train_ltn_xy[:, 1] + train_r_xy[:, 1]
    total_test_loss = test_xy[:, 1] + test_ltn_xy[:, 1] + test_r_xy[:, 1]
    min_test_loss_idx = np.argmin(total_test_loss)
    
    total_train_xy = np.hstack((train_xy[:, 0].reshape(-1, 1), total_train_loss.reshape(-1, 1)))
    total_test_xy = np.hstack((test_xy[:, 0].reshape(-1, 1), total_test_loss.reshape(-1, 1)))
    
    if 'lego' in model_name:
        mean_max = 25000
        mean_min = 15000
    else:
        if 800000 in train_xy[:, 0]:
            mean_max = 800000
            mean_min = 750000
        elif 600000 in train_xy[:, 0]:
            mean_max = 600000
            mean_min = 550000
        elif 240000 in train_xy[:, 0]:
            mean_max = 240000
            mean_min = 200000
        else:
            mean_max = 110000
            mean_min = 100000
        
    
    for_mean_loss = total_test_xy[(total_test_xy[:, 0] <= mean_max) & (total_test_xy[:, 0] >= mean_min)]
    
    plt.plot(train_xy[:, 0], total_train_loss, 'b', label = 'train')
    plt.plot(test_xy[:, 0], total_test_loss, 'g', label = 'test')
    plt.legend()
    plt.grid()
    
    if 'trainable' in model_name:
        title = model_name.split('_unconditional')[0]
        title = f'{title.split("_LE")[1]}'
    else:
        title = model_name.split('_unconditional')[0]
        title = f'VDM{title.split("_VDM")[1]}'
    title = f'{title}_total\n min test loss: {total_test_xy[:, 1][min_test_loss_idx]} at step: {total_test_xy[:, 0][min_test_loss_idx]}\n mean test loss {mean_min} to {mean_max}: {for_mean_loss[:, 1].mean()}'
    plt.title(title, fontsize = 8)
    
    if plot_dir == '':
        plt.show()
    else:        
        plot_file_name = f'{model_name}_train_test_total_loss.png'
        sample_path = os.path.join(plot_dir, plot_file_name)
        if not os.path.exists(sample_path):
            plt.savefig(sample_path, dpi = 300)
    
    plt.clf()
    
    for i in range(len(zoom_min)):        
        plt.xlim([zoom_min[i], zoom_max[i]])
        
        train_min, train_max, train_min_step = get_min_max_y(
            total_train_xy,
            zoom_min[i], zoom_max[i])
        test_min, test_max, test_min_step = get_min_max_y(
            total_test_xy,
            zoom_min[i], zoom_max[i])        
        y_max = np.maximum(train_max,test_max)
        y_min = np.minimum(train_min,test_min)
        
        if 'lego' in model_name:
            if 'larger' in model_name:
                if test_max < 1.5:
                    y_max = 1.5
            else:
                if test_max < 1.75:
                    y_max = 1.75                  
            
            if test_min > 0.9:
                y_min = 0.9
        elif 'mnist' in model_name:
            if test_max < 1.1:
                y_max = 1.1  
            if test_max > 0.5:
                y_min = 0.5
        else:
            if test_max < 3.5:
                y_max = 3.5  
            if test_max > 2.6:
                y_min = 2.6
        
        plt.ylim([y_min - 0.01, y_max + 0.01])# TODO: change this 
        plt.plot(train_xy[:, 0], total_train_loss, 'b', label = 'train')
        plt.plot(test_xy[:, 0], total_test_loss, 'g', label = 'test')
        plt.legend()
        plt.grid()
        
        if 'trainable' in model_name:
            title = model_name.split('_unconditional')[0]
            title = f'{title.split("_LE")[1]}'
        else:
            title = model_name.split('_unconditional')[0]
            title = f'VDM{title.split("_VDM")[1]}'
                
        title = f'{title}_total\n min test loss: {test_min} at step: {test_min_step}\n mean test loss {mean_min} to {mean_max}: {for_mean_loss[:, 1].mean()}'
        plt.title(title, fontsize = 8)
        
        if plot_dir == '':
            plt.show()
        else:        
            plot_file_name = f'{model_name}_train_test_total_loss_zoom_{zoom_min[i]}_{zoom_max[i]}.png'
            sample_path = os.path.join(plot_dir, plot_file_name)
            if not os.path.exists(sample_path):
                plt.savefig(sample_path, dpi = 300)
        
        plt.clf()
    
    
    
    if 't_diff_loss' in json_dict_metrics[0]['train']:
        train_t_diff_losses = {k: np.array(
            json_dict_metrics[k]['train']['t_diff_loss'])  
                        for k in json_dict_metrics}
    
        test_t_diff_losses = {k: np.array( 
             json_dict_metrics[k]['eval']['t_diff_loss'])  
                        for k in json_dict_metrics if 'eval' in json_dict_metrics[k]}
    
        to_array = lambda t: [t[0]] + t[1].tolist()
        
        train_sorted = sorted(train_t_diff_losses.items()) # sorted by key, return a list of tuples
        train_xy = np.array(list(map(to_array, train_sorted)))
        
        
        test_sorted = sorted(test_t_diff_losses.items()) # sorted by key, return a list of tuples
        test_xy = np.array(list(map(to_array, test_sorted)))
        
        plt.plot(train_xy[:, 0], train_xy[:, 1:], 'b', label = ['train', '', '', ''])
        
        plt.plot(test_xy[:, 0], test_xy[:, 1:], 'g', label = ['test', '', '', ''])
        plt.grid()
            
        plt.legend()
        if 'trainable' in model_name:
            title = model_name.split('_unconditional')[0]
            title = f'{title.split("_LE")[1]}_t_diff'
        else:
            title = model_name.split('_unconditional')[0]
            title = f'VDM{title.split("_VDM")[1]}_t_diff'
        plt.title(title, fontsize = 8)
        
        if plot_dir == '':
            plt.show()
        else:        
            plot_file_name = f'{model_name}_train_test_t_diff_loss.png'
            sample_path = os.path.join(plot_dir, plot_file_name)
            if not os.path.exists(sample_path):
                plt.savefig(sample_path, dpi = 300)
        
        plt.clf()
        
    
    
    

def plot_bpd_latent(load_from_folder: str, load_from_file: str, plot_dir: str, zoom_min: List[int] = [200000], zoom_max: List[int] = [300000]):
    load_from_path = os.path.join(load_from_folder, load_from_file)
    json_metrics = sljson.load_json(load_from_path)
    model_name = load_from_file.split('.')[0]

    json_dict_metrics = {int(k):elm[k] for elm in json_metrics for k in elm}
    
    if 'LEINV' in model_name and 'SPLIT' in model_name:
        #TODO: Make more general?
        rescale_to_bpd = 1./(jnp.prod(jnp.array((32,32,3))) * jnp.log(2.))
        
        train_diff_losses = {k: np.array(
            [json_dict_metrics[k]['train']['inv_loss']]) * rescale_to_bpd
                        for k in json_dict_metrics}

        test_diff_losses = {k: np.array( 
             [json_dict_metrics[k]['eval']['inv_loss']]) * rescale_to_bpd  
                        for k in json_dict_metrics if 'eval' in json_dict_metrics[k]}
    else:        
        train_diff_losses = {k: np.array(
            [json_dict_metrics[k]['train']['bpd_diff']])  
                        for k in json_dict_metrics}
    
        test_diff_losses = {k: np.array( 
             [json_dict_metrics[k]['eval']['bpd_diff']])  
                        for k in json_dict_metrics if 'eval' in json_dict_metrics[k]}


    train_sorted = sorted(train_diff_losses.items()) # sorted by key, return a list of tuples
    train_x, train_y = zip(*train_sorted) # unpack a list of pairs into two tuples

    plt.plot(train_x, train_y, 'b', label = 'train')

    test_sorted = sorted(test_diff_losses.items()) # sorted by key, return a list of tuples
    test_x, test_y = zip(*test_sorted) # unpack a list of pairs into two tuples

    plt.plot(test_x, test_y, 'g', label = 'test')
    plt.legend()
    if 'trainable' in model_name:
        title = model_name.split('_unconditional')[0]
        title = title.split('_LE')[1]
    else:
        title = model_name.split('_unconditional')[0]
        title = f'VDM{title.split("_VDM")[1]}'
    plt.title(title, fontsize = 8)
    
    if plot_dir == '':
        plt.show()
    else:        
        plot_file_name = f'{model_name}_train_test_diff_loss.png'
        sample_path = os.path.join(plot_dir, plot_file_name)
        if not os.path.exists(sample_path):
            plt.savefig(sample_path, dpi = 300)
    
    plt.clf()
    # Zoomed in plots 
    if len(zoom_min) == 0:
        if 100000 in train_diff_losses:
            zoom_min.append(50000)
            zoom_max.append(100000)
        if 170000 in train_diff_losses:
            zoom_min.append(70000)
            zoom_max.append(170000)
        if 250000 in train_diff_losses:
            zoom_min.append(70000)
            zoom_max.append(250000)
        if 300000 in train_diff_losses:            
            zoom_min.append(200000)
            zoom_max.append(300000)
        if 500000 in train_diff_losses:
            zoom_min.append(400000)
            zoom_max.append(500000)
        if 600000 in train_diff_losses:
            zoom_min.append(500000)
            zoom_max.append(600000)
        if 800000 in train_diff_losses:
            zoom_min.append(500000)
            zoom_max.append(800000)
        if 1000000 in train_diff_losses:
            zoom_min.append(900000)
            zoom_max.append(1000000)
            zoom_min.append(100000)
            zoom_max.append(1000000)
        if 1200000 in train_diff_losses:            
            zoom_min.append(100000)
            zoom_max.append(1200000)
        
    
    for i in range(len(zoom_min)):        
        plt.xlim([zoom_min[i], zoom_max[i]])
        
        train_min, train_max, train_min_step = get_min_max_y(
            train_diff_losses,
            zoom_min[i], zoom_max[i])
        test_min, test_max, test_min_step = get_min_max_y(
            test_diff_losses,
            zoom_min[i], zoom_max[i])        
        y_max = np.maximum(train_max,test_max)
        y_min = np.minimum(train_min,test_min)
        
        if 'lego' in model_name:
            if test_max < 1.5:
                y_max = 1.5  
            if test_max > 1.0:
                y_min = 1.0
        elif 'mnist' in model_name:
            if test_max < 1.1:
                y_max = 1.1  
            if test_max > 0.5:
                y_min = 0.5
        else:
            if test_max < 3.5:
                y_max = 3.5  
            if test_max > 2.6:
                y_min = 2.6
        plt.ylim([y_min - 0.01, y_max + 0.01])# TODO: change this 
        plt.plot(train_x, train_y, 'b', label = 'train')
        plt.plot(test_x, test_y, 'g', label = 'test')
        plt.legend()
        plt.grid()
        plt.title(title, fontsize = 8)
        
        if plot_dir == '':
            plt.show()
        else:        
            plot_file_name = f'{model_name}_train_test_diff_loss_zoom_{zoom_min[i]}_{zoom_max[i]}.png'
            sample_path = os.path.join(plot_dir, plot_file_name)
            if not os.path.exists(sample_path):
                plt.savefig(sample_path, dpi = 300)
        plt.clf()
        


def get_min_max_y(losses: NDArray, zoom_min: int, zoom_max: int):
    zoom_losses = losses[(losses[:, 0] <= zoom_max) & (losses[:, 0] >= zoom_min)]
    min_idx = np.argmin(zoom_losses[:, 1])
    min_step = zoom_losses[min_idx][0]
    return zoom_losses[min_idx][1], zoom_losses[:, 1:].max(), min_step


def plot_x_t_grad(load_from_folder: str, load_from_file: str, plot_dir: str, 
                  zoom_min: List[int] = [200000], zoom_max: List[int] = [300000]):
    
    load_from_path = os.path.join(load_from_folder, load_from_file)
    json_metrics = sljson.load_json(load_from_path)
    model_name = load_from_file.split('.')[0]

    json_dict_metrics = {int(k):elm[k] for elm in json_metrics for k in elm}
    
    train_min_mean_max_grad = {k: np.array(
        json_dict_metrics[k]['train']['x_t_grad_min_mean_max'])  
                    for k in json_dict_metrics}
    
    to_array = lambda t: [t[0]] + t[1].tolist()
    
    grad_sorted = sorted(train_min_mean_max_grad.items()) # sorted by key, return a list of tuples
    grad_xy = np.array(list(map(to_array, grad_sorted)))
    
    plt.plot(grad_xy[:, 0], grad_xy[:, 1:])
    title = model_name.split('_unconditional')[0]
    title = title.split('_LE')[1]
    plt.title(title, fontsize = 8)
    
    if plot_dir == '':
        plt.show()
    else:        
        plot_file_name = f'{model_name}_x_t_grad_min_mean_max.png'
        sample_path = os.path.join(plot_dir, plot_file_name)
        if not os.path.exists(sample_path):
            plt.savefig(sample_path, dpi = 300)
    
    plt.clf()
    
    # Zoomed in plots 
    if len(zoom_min) == 0:
        if 'lego' in model_name:            
            if 20000 in train_min_mean_max_grad:
                 zoom_min.append(5000)
                 zoom_max.append(20000)            
            if 60000 in train_min_mean_max_grad:
                zoom_min.append(5000)
                zoom_max.append(60000)
        if 100000 in train_min_mean_max_grad:
            zoom_min.append(50000)
            zoom_max.append(100000)
        if 170000 in train_min_mean_max_grad:
            zoom_min.append(70000)
            zoom_max.append(170000)
        if 300000 in train_min_mean_max_grad:
            zoom_min.append(200000)
            zoom_max.append(300000)
        if 500000 in train_min_mean_max_grad:
            zoom_min.append(400000)
            zoom_max.append(500000)
        if 600000 in train_min_mean_max_grad:
            zoom_min.append(500000)
            zoom_max.append(600000)
        if 800000 in train_min_mean_max_grad:
            zoom_min.append(500000)
            zoom_max.append(800000)
        if 1000000 in train_min_mean_max_grad:
            zoom_min.append(900000)
            zoom_max.append(1000000)
            zoom_min.append(100000)
            zoom_max.append(1000000)
        if 1200000 in train_min_mean_max_grad:            
            zoom_min.append(100000)
            zoom_max.append(1200000)
    
    for i in range(len(zoom_min)):        
        plt.xlim([zoom_min[i], zoom_max[i]])
        
        train_min, train_max, train_min_step = get_min_max_y(
            grad_xy,
            zoom_min[i], zoom_max[i])
        y_max = train_max
        if train_max < 1.2:
            y_max = 1.2       
        y_min = train_min
        if train_min > -1.2:
            y_min = -1.2
        plt.ylim([y_min - 0.01, y_max + 0.01])# TODO: change this 
        plt.plot(grad_xy[:, 0], grad_xy[:, 1:])    
        plt.grid()
        plt.title(title, fontsize = 8)
        
        if plot_dir == '':
            plt.show()
        else:        
            plot_file_name = f'{model_name}_x_t_grad_min_mean_max_zoom_{zoom_min[i]}_{zoom_max[i]}.png'
            sample_path = os.path.join(plot_dir, plot_file_name)
            if not os.path.exists(sample_path):
                plt.savefig(sample_path, dpi = 300)
            plt.clf()
    
    


def plot_loss_folder(folder: str, plot_dir: str, pattern: str = ''):
    loss_files = [file 
                    for file in os.listdir(folder)
                    if re.match(f'.*{pattern}.*_train_metrics_tidy.json', file)]
    
    for file in loss_files:
        plot_losses(folder, file, plot_dir, zoom_min=[], zoom_max=[])
        
        
def plot_x_t_grad_folder(folder: str, plot_dir: str, pattern: str = ''):
    loss_files = [file 
                    for file in os.listdir(folder)
                    if re.match(f'.*{pattern}.*_train_metrics_tidy.json', file)]
    
    for file in loss_files:
        plot_x_t_grad(folder, file, plot_dir, zoom_min=[], zoom_max=[])


def loss_dict_to_array(json_dict_metrics: Dict, train_eval: str, loss_type: str):
    type_losses = {k: np.array(
        json_dict_metrics[k][train_eval][loss_type])  
                    for k in json_dict_metrics
                    if train_eval in json_dict_metrics[k]}

    losses_sorted = sorted(type_losses.items()) # sorted by key, return a list of tuples
    loss_xy = np.array(losses_sorted)
    
    return loss_xy


def plot_loss_with_zooms(train_xy: NDArray, test_xy: NDArray,
                         loss_name: str,
                         model_name: str, plot_dir: str,
                         zoom_min: List[int], zoom_max: List[int]):
    if 'lego' in model_name:
        mean_max = 25000
        mean_min = 15000
    else:
        mean_max = 240000
        mean_min = 200000
    
    
    for_mean_loss = test_xy[(test_xy[:, 0] <= mean_max) & (test_xy[:, 0] >= mean_min)]
    
    min_test_diff_loss_idx = np.argmin(test_xy[:, 1])
    
    if loss_name == 'recon':
        y_max = 1.5
        y_min = 0.0
        if test_xy[:, 1].max() < 0.2:
            y_max = 0.2
        plt.ylim([y_min, y_max])
    
    plt.plot(train_xy[:, 0], train_xy[:, 1], 'b', label = 'train')
    plt.plot(test_xy[:, 0], test_xy[:, 1], 'g', label = 'test')
    plt.grid() 
    plt.legend()
    if 'trainable' in model_name:
        title = model_name.split('_unconditional')[0]
        base_title = f'{title.split("_LE")[1]}'
    else:
        title = model_name.split('_unconditional')[0]
        base_title = f'VDM{title.split("_VDM")[1]}'
    title = f'{base_title}_{loss_name}\n min test loss: {test_xy[:, 1][min_test_diff_loss_idx]} at step: {test_xy[:, 0][min_test_diff_loss_idx]}\n mean test loss {mean_min} to {mean_max}: {for_mean_loss[:, 1].mean()}'
    plt.title(title, fontsize = 8)
    
    if plot_dir == '':
        plt.show()
    else:        
        plot_file_name = f'{model_name}_train_test_{loss_name}_loss.png'
        sample_path = os.path.join(plot_dir, plot_file_name)
        if not os.path.exists(sample_path):
            plt.savefig(sample_path, dpi = 300)
    
    plt.clf()
    # Zoomed in plots 
    if len(zoom_min) == 0: 
        if 'lego' in model_name:            
            if 20000 in train_xy[:, 0]:
                 zoom_min.append(5000)
                 zoom_max.append(20000)            
            if 60000 in train_xy[:, 0]:
                zoom_min.append(5000)
                zoom_max.append(60000)
            
        if 100000 in train_xy[:, 0]:
            zoom_min.append(50000)
            zoom_max.append(100000)
        if 170000 in train_xy[:, 0]:
            zoom_min.append(70000)
            zoom_max.append(170000)
        if 250000 in train_xy[:, 0]:
            zoom_min.append(70000)
            zoom_max.append(250000)
        if 300000 in train_xy[:, 0]:
            zoom_min.append(200000)
            zoom_max.append(300000)
        if 500000 in train_xy[:, 0]:
            zoom_min.append(400000)
            zoom_max.append(500000)
        if 600000 in train_xy[:, 0]:
            zoom_min.append(500000)
            zoom_max.append(600000)
        if 800000 in train_xy[:, 0]:
            zoom_min.append(500000)
            zoom_max.append(800000)
        if 1000000 in train_xy[:, 0]:
            zoom_min.append(900000)
            zoom_max.append(1000000)
            zoom_min.append(100000)
            zoom_max.append(1000000)
        if 1200000 in train_xy[:, 0]:            
            zoom_min.append(100000)
            zoom_max.append(1200000)
    
    for i in range(len(zoom_min)):        
        plt.xlim([zoom_min[i], zoom_max[i]])
        
        train_min, train_max, train_min_step = get_min_max_y(
            train_xy,
            zoom_min[i], zoom_max[i])
        test_min, test_max, test_min_step = get_min_max_y(
            test_xy,
            zoom_min[i], zoom_max[i])        
        y_max = np.maximum(train_max,test_max)
        y_min = np.minimum(train_min,test_min)
        
        if loss_name == 'diff':
            if 'lego' in model_name:
                if 'larger' in model_name:
                    if test_max < 1.5:
                        y_max = 1.5
                else:
                    if test_max < 1.75:
                        y_max = 1.75                  
                
                if test_min > 0.9:
                    y_min = 0.9
            elif 'mnist' in model_name:
                if test_max < 1.1:
                    y_max = 1.1  
                if test_max > 0.5:
                    y_min = 0.5
            else:
                if test_max < 3.5:
                    y_max = 3.5  
                if test_max > 2.6:
                    y_min = 2.6
        if loss_name == 'recon':
            y_max = 1.5
            y_min = 0.0
        
        plt.ylim([y_min - 0.01, y_max + 0.01])# TODO: change this 
        plt.plot(train_xy[:, 0], train_xy[:, 1], 'b', label = 'train')
        plt.plot(test_xy[:, 0], test_xy[:, 1], 'g', label = 'test')
        plt.legend()
        plt.grid()
        title = f'{base_title}_{loss_name}\n min test loss: {test_min} at step: {test_min_step}\n mean test loss {mean_min} to {mean_max}: {for_mean_loss[:, 1].mean()}'
        plt.title(title, fontsize = 8)
        
        if plot_dir == '':
            plt.show()
        else:        
            plot_file_name = f'{model_name}_train_test_{loss_name}_loss_zoom_{zoom_min[i]}_{zoom_max[i]}.png'
            sample_path = os.path.join(plot_dir, plot_file_name)
            if not os.path.exists(sample_path):
                plt.savefig(sample_path, dpi = 300)
        plt.clf()


def tidy_loss_folder(folder: str, pattern: str = ''):
    loss_files = [file 
                    for file in os.listdir(folder)
                    if re.match(f'.*{pattern}.*_train_metrics.json', file)]
    
    for file in loss_files:
        tidy_loss_log(folder, file)


# Remove duplicates resulting from resuming training.
def tidy_loss_log(folder_path: str, file_name: str):
    load_from_path = os.path.join(folder_path, file_name)
    json_metrics = sljson.load_json(load_from_path)
    
    keys = set()
    new_metrics = []
    
    for i in reversed(range(len(json_metrics))):
        current_elm  = json_metrics[i]
        current_key = next(iter(current_elm.keys()))
        
        if current_key not in keys:
            if len(keys) == 0 or (int(current_key) < np.min([*keys])):
                keys.add(int(current_key))
                new_metrics.append(current_elm)
        
    new_metrics.reverse()
    
    sljson.save_as_json(new_metrics, load_from_path.replace('.json', '_tidy.json'))
    

# Example pattern: 'LE_EPSGAMMA_SIGMAENC_fixed_-12_8_2_0_unet_4_trainable_unet_2_maxpool_lego_larger_unconditional_.*_64_clip_2_0'
def get_combined_losses(folder_path: str, pattern: str, allow_single_file: bool = False):    
    loss_files = [file 
                    for file in os.listdir(folder_path)
                    if re.match(f'.*{pattern}.*_metrics_tidy.json', file)]
    
    if len(loss_files) == 0:
        raise ValueError(f'No loss files found with pattern: {pattern}')
    
    if not allow_single_file and len(loss_files) < 3:
        return {}
    
    keys = set()
    dicts = []
    
    for file in loss_files:
        load_from_path = os.path.join(folder_path, file)
        json_metrics = sljson.load_json(load_from_path)        

        json_dict_metrics = {int(k):elm[k] for elm in json_metrics for k in elm}
        current_keys = set(json_dict_metrics.keys())
        
        if not keys:
            keys = current_keys
        
        keys = keys.intersection(current_keys)
        dicts.append(json_dict_metrics)
    
    # Remove keys which should not be used
    for d in dicts:
        d_keys = set(d)
        for k in d_keys - keys:
            del d[k]
    
    combined_losses_dict = {}
    
    loss_names = ['bpd_diff', 'bpd_recon', 'bpd_latent']
    train_eval = ['train', 'eval']
    
    for loss_name in loss_names:
        combined_losses_dict[loss_name] = {}
        for t_e in train_eval:
            if t_e == 'train':
                combined_losses_dict[loss_name][t_e] = np.zeros((len(keys), len(dicts) + 1))
            else:
                ex_d = dicts[0]
                eval_keys = [k for k in ex_d if 'eval' in ex_d[k]]
                combined_losses_dict[loss_name][t_e] = np.zeros((len(eval_keys), len(dicts) + 1))
            
            for i, d in enumerate(dicts):
                xy = loss_dict_to_array(d, t_e, loss_name)
                combined_losses_dict[loss_name][t_e][:, i + 1] = xy[:, 1]
                
                if combined_losses_dict[loss_name][t_e][:, 0].sum() == 0:
                    combined_losses_dict[loss_name][t_e][:, 0] = xy[:, 0]
    
    return combined_losses_dict


def get_total_from_combined_loss_dict(comb_d: Dict, train_eval: str):
    loss_names = ['bpd_diff', 'bpd_recon', 'bpd_latent']
    
    one_loss = comb_d[loss_names[0]][train_eval]
    one_loss_shape = one_loss.shape
        
    total_losses = np.zeros(one_loss_shape)
    total_losses[:, 0] = one_loss[:, 0]
    for i in range(len(loss_names)):
        total_losses[:, 1:] = total_losses[:, 1:] + comb_d[loss_names[i]][train_eval][:, 1:]
        
    return total_losses
    
    
def plot_VDM_baseline_vs_single_model(
        folder_path: str, 
        plot_dir: str, 
        dataset: str, 
        edm_gamma_range: str = '-13.3 to 5.0',
        vdm_gamma_range: str = '-13.3 to 5.0',
        edm_model_type: str = 'S3VA',
        vdm_model_type: str = 'VDM2',
        enc_model: str = 'unet_2',
        edm_extras: str = 'm1_to_1'):
    
    edm_g_range_file = edm_gamma_range.replace(' to ', '_').replace('.', '_')
    vdm_g_range_file = vdm_gamma_range.replace(' to ', '_').replace('.', '_')        
        
    edm_pattern = get_regex_patterns(
        [edm_gamma_range], 
        dataset,
        edm_model_type,
        enc_model = enc_model,
        edm_extras = edm_extras)[0]
    vdm_pattern = get_regex_patterns(
        [vdm_gamma_range],
        dataset,
        vdm_model_type)[0]
        
    combined_vdm = get_combined_losses(folder_path, vdm_pattern)
    combined_edm = get_combined_losses(folder_path, edm_pattern, 
                                       allow_single_file = True)
    
    total_test_loss_edm = get_total_from_combined_loss_dict(combined_edm, 'eval')
    total_test_loss_vdm = get_total_from_combined_loss_dict(combined_vdm, 'eval')
        
    plot_lossses_vs_loss_zoom(
        total_test_loss_edm, 
        total_test_loss_vdm,
        dataset,
        'total_eval', 
        edm_gamma_range,
        vdm_gamma_range)
    
    if edm_extras != '':
        edm_extras = edm_extras + '_'
        
    plot_file_name = f'{edm_model_type}_{edm_g_range_file}_{edm_extras}vs_{vdm_model_type}_{vdm_g_range_file}_{dataset}_total_loss_test.png'
    plot_path = os.path.join(plot_dir, plot_file_name)
    if not os.path.exists(plot_path):
        plt.savefig(plot_path, dpi = 300)
    plt.clf()
    
    loss_names = ['bpd_diff', 'bpd_recon', 'bpd_latent']
    train_eval = ['train', 'eval']
    
    for name in loss_names:      
        for t_e in train_eval:
            
            loss_edm = combined_edm[name][t_e]
            loss_vdm = combined_vdm[name][t_e]
        
            loss_type = f'{name}_{t_e}'
            plot_lossses_vs_loss_zoom(
                loss_edm, 
                loss_vdm,
                dataset,
                loss_type, 
                edm_gamma_range,
                vdm_gamma_range)
            plot_file_name = f'{edm_model_type}_{edm_g_range_file}_{edm_extras}vs_{vdm_model_type}_{vdm_g_range_file}_{dataset}_{loss_type}.png'
            plot_path = os.path.join(plot_dir, plot_file_name)
            if not os.path.exists(plot_path):
                plt.savefig(plot_path, dpi = 300)
            plt.clf()
    


def plot_lossses_vs_loss_zoom(
        loss_edm: NDArray, 
        loss_vdm: NDArray,
        dataset: str,
        loss_type: str,
        edm_gamma: str,
        vdm_gamma: str):
    zoom_min = 1
    if dataset == 'lego_larger':
        if '_diff_' in loss_type or 'total' in loss_type:
            zoom_min = 2000
        
        zoom_max = 60000
    elif dataset == 'mnist':
        zoom_max = 600000
    elif dataset == 'cifar10':
        zoom_max = 450000
    else:
        raise ValueError(f'Dataset not implemented: {dataset}')
    
    zoom_edm = loss_edm[(loss_edm[:, 0] <= zoom_max) & (loss_edm[:, 0] >= zoom_min)]
    zoom_vdm = loss_vdm[(loss_vdm[:, 0] <= zoom_max) & (loss_vdm[:, 0] >= zoom_min)]
    
    # Colour tuples darker to lighter
    green_tuple = ('#006d2c', '#31a354', '#74c476', '#a1d99b')
    blue_tuple = ('#08519c', '#3182bd', '#6baed6', '#9ecae1')
    purple_tuple = ('#54278f', '#756bb1', '#9e9ac8', '#bcbddc')
    
    if '-13.3' in vdm_gamma and '5.0' in vdm_gamma:
        vdm_tuple = (blue_tuple[0], blue_tuple[2])
    else:
        vdm_tuple = (blue_tuple[1], blue_tuple[3])
    
    if '-13.3' in edm_gamma and '5.0' in edm_gamma:
        edm_tuple = (green_tuple[0], green_tuple[2])
    else:
        edm_tuple = (green_tuple[1], green_tuple[3])
    
    plot_losses_with_standard_error_of_mean(zoom_vdm, f'vdm {vdm_gamma}', vdm_tuple) 
        
    plot_loss(zoom_edm, f'edm {edm_gamma}', edm_tuple)
        
        
    plt.xlim([1, zoom_max])
    
    if dataset == 'lego_larger':
        if 'recon' in loss_type:
            plt.ylim([0.005, 0.055])
        elif 'latent' in loss_type:
            plt.ylim([0.0, 0.064])
        elif 'total' in loss_type:
            plt.ylim([1.15, 1.71])
    elif dataset == 'mnist':
        if 'total' in loss_type:
            plt.ylim([0.5, 1.2])
    elif dataset == 'cifar10':
        if 'recon' in loss_type:
            plt.ylim([0.005, 0.02])
        elif 'latent' in loss_type:
            plt.ylim([0.0, 0.01])
        elif 'total' in loss_type:
            if jnp.max(zoom_edm[25:, 1:]) < 3.3:
                plt.ylim([2.9, 3.3])
    else:
        raise ValueError(f'Dataset not implemented: {dataset}')
    
    
    plt.grid()
    plt.legend()
    plt.xlabel('Training step')
    plt.ylabel('Loss in BPD')
    
    if zoom_max > 200000:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.title(loss_type)
    #plt.show()     
    



def plot_model_vs_model_losses(
        folder_path: str, 
        plot_dir: str, 
        dataset: str,
        gamma_range1: str = '-13.3 to 5.0',
        gamma_range2: str = '-13.3 to 5.0',
        model_type1: str = 'S3VA',
        model_type2: str = 'VDM2',
        noise_schedule_type1: str = 'fixed',
        noise_schedule_type2: str = 'fixed',
        enc_models: Tuple[str, str] = ('unet_2', 'unet_2'),
        edm_extras: Tuple[str, str] = ('m1_to_1', 'm1_to_1'),
        batch_size: int = 128,
        diff_size: int = 8,
        plot_all: bool = False):
    suffix = ''
    if plot_all:
        suffix = '_all'
    
    if 'VDM' in model_type1:
        model1_pattern = get_regex_patterns(
            [gamma_range1],
            dataset,
            model_type1,
            noise_schedule_type = noise_schedule_type1,
            batch_size = batch_size,
            diff_size = diff_size)[0]
    else:
        model1_pattern = get_regex_patterns(
            [gamma_range1], 
            dataset,
            model_type1,
            noise_schedule_type = noise_schedule_type1,
            enc_model = enc_models[0],
            edm_extras = edm_extras[0],
            batch_size = batch_size,
            diff_size = diff_size)[0]
    
    if 'VDM' in model_type2:
        model2_pattern = get_regex_patterns(
            [gamma_range2],
            dataset,
            model_type2,
            noise_schedule_type = noise_schedule_type2,
            batch_size = batch_size,
            diff_size = diff_size)[0]
    else:
        model2_pattern = get_regex_patterns(
            [gamma_range1], 
            dataset,
            model_type2,
            noise_schedule_type = noise_schedule_type2,
            enc_model = enc_models[1],
            edm_extras = edm_extras[1],
            batch_size = batch_size,
            diff_size = diff_size)[0]
         
    g_range_file1 = gamma_range1.replace(' to ', '_').replace('.', '_')
    g_range_file2 = gamma_range2.replace(' to ', '_').replace('.', '_')        
    
    combined_1 = get_combined_losses(folder_path, model1_pattern, allow_single_file = True)
    combined_2 = get_combined_losses(folder_path, model2_pattern, allow_single_file = True)
    
    total_test_loss_1 = get_total_from_combined_loss_dict(combined_1, 'eval')
    total_test_loss_2 = get_total_from_combined_loss_dict(combined_2, 'eval')
    
    if dataset == 'cifar10':
        if diff_size == 32:
            zoom_max = 9000000
        else:
            zoom_max = 2000000
    elif dataset == 'mnist':
        zoom_max = 2200000
    elif dataset == 'imagenet32':
        zoom_max = 2000000
    
    plot_loss_vs_loss_zoom(
        total_test_loss_1, 
        total_test_loss_2,
        dataset,
        'total_eval', 
        (gamma_range1, gamma_range2),
        (model_type1, model_type2),
        'eval',
        plot_all,
        zoom_max)
    
    plot_file_name = f'{model_type1}_{diff_size}{enc_models[0]}_{noise_schedule_type1}_{g_range_file1}_vs_{model_type2}_{diff_size}{enc_models[1]}_{noise_schedule_type2}_{g_range_file2}_{dataset}_total_loss_test{suffix}.png'
    plot_path = os.path.join(plot_dir, plot_file_name)
    if not os.path.exists(plot_path):
        plt.savefig(plot_path, dpi = 300)
    plt.clf()
    
    loss_names = ['bpd_diff', 'bpd_recon', 'bpd_latent']
    train_eval = ['train', 'eval']
    
    for name in loss_names:      
        for t_e in train_eval:
            
            loss_1 = combined_1[name][t_e]
            loss_2 = combined_2[name][t_e]
            
            loss_type = f'{name}_{t_e}'
            plot_loss_vs_loss_zoom(
                loss_1, 
                loss_2,
                dataset,
                loss_type, 
                (gamma_range1, gamma_range2),
                (model_type1, model_type2),
                t_e,
                plot_all,
                zoom_max)
            plot_file_name = f'{model_type1}_{diff_size}{enc_models[0]}_{noise_schedule_type1}_{g_range_file1}_vs_{model_type2}_{diff_size}{enc_models[1]}_{noise_schedule_type2}_{g_range_file2}_{dataset}_{loss_type}{suffix}.png'
            plot_path = os.path.join(plot_dir, plot_file_name)
            if not os.path.exists(plot_path):
                plt.savefig(plot_path, dpi = 300)
            plt.clf()

    

def plot_loss_vs_loss_zoom(
        loss_1: NDArray, 
        loss_2: NDArray,
        dataset: str,
        loss_type: str,
        gammas: Tuple[str, str],
        model_names: Tuple[str, str],
        train_eval: str,
        plot_all: bool = False,
        zoom_max: int = 800000):
    zoom_min = 1
    if dataset == 'lego_larger':
        if '_diff_' in loss_type or 'total' in loss_type:
            zoom_min = 2000
        
        zoom_max = 60000
    elif dataset == 'cifar10' or dataset == 'imagenet32':
        if '_diff_' in loss_type or 'total' in loss_type:
            zoom_min = 2000  
    elif dataset == 'mnist':
        if '_diff_' in loss_type or 'total' in loss_type:
            zoom_min = 50000        
    else:
        raise ValueError(f'Dataset not implemented: {dataset}')
    
    zoom_edm = loss_1[(loss_1[:, 0] <= zoom_max) & (loss_1[:, 0] >= zoom_min)]
    zoom_vdm = loss_2[(loss_2[:, 0] <= zoom_max) & (loss_2[:, 0] >= zoom_min)]
    
    # Colour tuples darker to lighter
    green_tuple = ('#006d2c', '#31a354', '#74c476', '#a1d99b')
    blue_tuple = ('#08519c', '#3182bd', '#6baed6', '#9ecae1')
    purple_tuple = ('#54278f', '#756bb1', '#9e9ac8', '#bcbddc')
       
    
    if '-13.3' in gammas[0] and '5.0' in gammas[0]:
        tuple_1 = (green_tuple[0], green_tuple[2])
    else:
        tuple_1 = (green_tuple[1], green_tuple[3])
        
    if '-13.3' in gammas[1] and '5.0' in gammas[1]:
        tuple_2 = (blue_tuple[0], blue_tuple[2])
    else:
        tuple_2 = (blue_tuple[1], blue_tuple[3])
    
    
    if plot_all:
        plot_all_losses_with_mean(zoom_edm, f'{model_names[0]} {gammas[0]}', tuple_1)
        plot_all_losses_with_mean(zoom_vdm, f'{model_names[1]} {gammas[1]}', tuple_2)
    else:        
        plot_losses_with_standard_deviation_of_mean(zoom_edm, f'{model_names[0]} {gammas[0]}', tuple_1)
        plot_losses_with_standard_deviation_of_mean(zoom_vdm, f'{model_names[1]} {gammas[1]}', tuple_2)
        
    plt.xlim([1, zoom_max])
    
    if dataset == 'lego_larger':
        if 'recon' in loss_type:
            plt.ylim([0.005, 0.055])
        elif 'latent' in loss_type:
            plt.ylim([0.0, 0.064])
        elif 'total' in loss_type:
            plt.ylim([1.15, 1.71])
    elif dataset == 'cifar10':
        if 'recon' in loss_type:
            plt.ylim([0.0, 0.02])
        elif 'latent' in loss_type:
            if np.max(zoom_vdm[:, 1:]) > 0.002:
                plt.ylim([0.0, 0.004])
            else:
                plt.ylim([0.0, 0.0025])
        elif 'diff' in loss_type:
            if 'eval' in train_eval: 
                if np.min(zoom_edm[:, 1:]) > 2.76:
                    plt.ylim([2.75, 2.9])
                elif np.min(zoom_edm[:, 1:]) > 2.72:
                    plt.ylim([2.71, 2.8])
                else:
                    plt.ylim([2.6, 2.7])
            else:
                plt.ylim([2.2, 3.5])
        elif 'total' in loss_type:
            if 'eval' in train_eval: 
                if np.min(zoom_edm[:, 1:]) > 2.76:
                    plt.ylim([2.75, 2.9])
                elif np.min(zoom_edm[:, 1:]) > 2.72:
                    plt.ylim([2.71, 2.8])
                else:
                    plt.ylim([2.6, 2.7])
            else:
                plt.ylim([2.2, 3.5])
    elif dataset == 'imagenet32':
        if 'recon' in loss_type:
            plt.ylim([0.0, 0.02])
        elif 'latent' in loss_type:
            if np.max(zoom_vdm[:, 1:]) > 0.002:
                plt.ylim([0.0, 0.004])
            else:
                plt.ylim([0.0, 0.0025])
        elif 'diff' in loss_type:
            if 'eval' in train_eval: 
                if np.min(zoom_edm[:, 1:]) > 3.65:
                    plt.ylim([3.4, 3.85])
                else:
                    plt.ylim([3.4, 3.55])
            else:
                plt.ylim([3.0, 4.0])
        elif 'total' in loss_type:
            if 'eval' in train_eval: 
                if np.min(zoom_edm[:, 1:]) > 3.65:
                    plt.ylim([3.4, 3.85])
                else:
                    plt.ylim([3.4, 3.55])
            else:
                plt.ylim([3.0, 4.0])
                
    elif dataset == 'mnist':
        if 'recon' in loss_type:
            plt.ylim([0.0, 0.02])
        elif 'latent' in loss_type:
            if np.max(zoom_vdm[:, 1:]) > 0.005:
                plt.ylim([0.0, 0.007])
            else:
                plt.ylim([0.0, 0.0055])
        elif 'diff' in loss_type:
            plt.ylim([0.3, 0.5])
        elif 'total' in loss_type:
            plt.ylim([0.3, 0.5])
    else:
        raise ValueError(f'Dataset not implemented: {dataset}')
    
    
    plt.grid()
    plt.legend()
    plt.xlabel('Training step')
    plt.ylabel('Loss in BPD')
    
    if zoom_max > 200000:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #plt.title(loss_type)
    #plt.show()     
    
    
# Plot with standard error. standard colours are dark and light blue
# Other colour suggestion dark/light green: ('#006d2c', '#31a354', '#74c476')
# Other colour suggestion dark/light purple: ('#54278f', '#756bb1', '#9e9ac8')
def plot_losses_with_standard_error_of_mean(
        losses: NDArray, 
        label: str,
        colors: Tuple[str, str] = ('#08519c', '#6baed6')):
    mean, std, std_err = get_loss_mean_std_and_std_err(losses)
        
    x = losses[:, 0]
        
    plt.plot(x, mean, '-', color = colors[0], label = f'{label}', zorder = 20)
    
    plt.fill_between(x, mean-std_err, mean+std_err, color = colors[1], alpha=0.8, zorder = 15)


# Plot with standard deviation. standard colours are dark and light blue
# Other colour suggestion dark/light green: ('#006d2c', '#31a354', '#74c476')
# Other colour suggestion dark/light purple: ('#54278f', '#756bb1', '#9e9ac8')
def plot_losses_with_standard_deviation_of_mean(
        losses: NDArray, 
        label: str,
        colors: Tuple[str, str] = ('#08519c', '#6baed6')):
    mean, std, std_err = get_loss_mean_std_and_std_err(losses)
        
    x = losses[:, 0]
        
    plt.plot(x, mean, '-', color = colors[0], label = f'{label}', zorder = 20)
    
    plt.fill_between(x, mean-std, mean+std, color = colors[1], alpha=0.8, zorder = 15)



def plot_loss(
        losses: NDArray, 
        label: str,
        colors: Tuple[str, str] = ('#08519c', '#6baed6')):
            
    x = losses[:, 0]
    y = losses[:, 1]
        
    plt.plot(x, y, '-', color = colors[0], label = f'{label}', zorder = 20)
    

# Plot with standard deviation. standard colours are dark and light blue
# Other colour suggestion dark/light green: ('#006d2c', '#31a354', '#74c476')
def plot_all_losses_with_mean(
        losses: NDArray, 
        label: str,
        colors: Tuple[str, str] = ('#08519c', '#6baed6')):
    
    #std = np.std(losses[:, 1:], axis = 1)
    #std_err = std / np.sqrt(losses[:, 1:].shape[1] -1)
    mean = np.mean(losses[:, 1:], axis = 1)
    
    
    x = losses[:, 0]
    
    plt.plot(x, losses[:, 1], '-', color = colors[1], alpha=0.4, label = f'{label} seed', zorder = 10)
    
    for i in range(losses[:, 2:].shape[1]):
        plt.plot(x, losses[:, 2 + i], '-', color = colors[1], alpha=0.4, zorder = 10)    
    
    plt.plot(x, mean, '-', color = colors[0], label = f'{label}', zorder = 20)
    
    #plt.fill_between(x, mean-std_err, mean+std_err, color = colors[2], alpha=0.8, zorder = 15)
    

def plot_2_VDM_vs_EDM_losses(
        folder_path: str, 
        plot_dir: str, 
        dataset: str,        
        gamma_range1: str = '-12.8 to 2.0',
        gamma_range2: str = '-13.3 to 5.0',
        plot_all: bool = False):
    suffix = ''
    if plot_all:
        suffix = '_all'
    
    edm_patterns = get_regex_patterns(
        [gamma_range1, gamma_range2],
        dataset,
        'LE_EPSGAMMA_SIGMAENC')
    
    vdm_patterns = get_regex_patterns(
        [gamma_range1, gamma_range2],
        dataset,
        'VDM2')
    
    
    g_range_file1 = gamma_range1.replace(' to ', '_').replace('.', '_')
    g_range_file2 = gamma_range2.replace(' to ', '_').replace('.', '_')
    
    edm1_pattern = edm_patterns[0]
    edm2_pattern = edm_patterns[1]
    vdm1_pattern = vdm_patterns[0]
    vdm2_pattern = vdm_patterns[1]
    
    combined_edm1 = get_combined_losses(folder_path, edm1_pattern)
    combined_edm2 = get_combined_losses(folder_path, edm2_pattern)
    combined_vdm1 = get_combined_losses(folder_path, vdm1_pattern)
    combined_vdm2 = get_combined_losses(folder_path, vdm2_pattern)
    
    loss_names = ['bpd_recon', 'bpd_latent']
    train_eval = ['eval']
    
    for name in loss_names:      
        for t_e in train_eval:
            
            loss_edm1 = combined_edm1[name][t_e]
            if combined_edm2:
                loss_edm2 = combined_edm2[name][t_e]
            else:
                loss_edm2 = np.array([])
            loss_vdm1 = combined_vdm1[name][t_e]
            loss_vdm2 = combined_vdm2[name][t_e]
        
            loss_type = f'{name}_{t_e}'
            plot_2_loss_vs_loss_zoom(
                loss_edm1,
                loss_edm2,
                loss_vdm1,
                loss_vdm2,
                loss_type, 
                gamma_range1,
                gamma_range2,
                dataset,
                plot_all)
            plot_file_name = f'EDM_vs_VDM_{g_range_file1}_{g_range_file2}_{dataset}_{loss_type}{suffix}.png'
            plot_path = os.path.join(plot_dir, plot_file_name)
            if not os.path.exists(plot_path):
                plt.savefig(plot_path, dpi = 300)
            plt.clf()
    
    
def plot_2_loss_vs_loss_zoom(
        loss_edm1: NDArray, 
        loss_edm2: NDArray,
        loss_vdm1: NDArray, 
        loss_vdm2: NDArray, 
        loss_type: str,
        gamma1: str,
        gamma2: str,
        dataset: str,
        plot_all: bool):
    zoom_min = 1
    
    if dataset == 'lego_larger':
        if '_diff_' in loss_type or 'total' in loss_type:
            zoom_min = 2000
        
        zoom_max = 60000
    elif dataset == 'cifar10':
        zoom_max = 450000
    else:
        raise ValueError(f'Dataset not implemented: {dataset}')
    
    
    zoom_edm1 = loss_edm1[(loss_edm1[:, 0] <= zoom_max) & (loss_edm1[:, 0] >= zoom_min)]
    if loss_edm2.size > 0:
        zoom_edm2 = loss_edm2[(loss_edm2[:, 0] <= zoom_max) & (loss_edm2[:, 0] >= zoom_min)]
    else:
        zoom_edm2 = np.array([])
    zoom_vdm1 = loss_vdm1[(loss_vdm1[:, 0] <= zoom_max) & (loss_vdm1[:, 0] >= zoom_min)]
    zoom_vdm2 = loss_vdm2[(loss_vdm2[:, 0] <= zoom_max) & (loss_vdm2[:, 0] >= zoom_min)]
    
    # Colours, darker to lighter
    green_tuple = ('#006d2c', '#31a354', '#74c476', '#a1d99b')
    blue_tuple = ('#08519c', '#3182bd', '#6baed6', '#9ecae1')
    purple_tuple = ('#54278f', '#756bb1', '#9e9ac8', '#bcbddc')
    
    if plot_all:
        plot_all_losses_with_mean(zoom_edm1, f'edm {gamma1}', (green_tuple[1], green_tuple[3]))
        if zoom_edm2.size > 0:
            plot_all_losses_with_mean(zoom_edm2, f'edm {gamma2}', (green_tuple[0], green_tuple[2]))
        plot_all_losses_with_mean(zoom_vdm1, f'vdm {gamma1}', (blue_tuple[1], blue_tuple[3]))
        plot_all_losses_with_mean(zoom_vdm2, f'vdm {gamma2}', (blue_tuple[0], blue_tuple[2]))
    else:        
        plot_losses_with_standard_error_of_mean(zoom_edm1, f'edm {gamma1}', (green_tuple[1], green_tuple[3]))
        if zoom_edm2.size > 0:
            plot_losses_with_standard_error_of_mean(zoom_edm2, f'edm {gamma2}', (green_tuple[0], green_tuple[2]))
        plot_losses_with_standard_error_of_mean(zoom_vdm1, f'vdm {gamma1}', (blue_tuple[1], blue_tuple[3]))
        plot_losses_with_standard_error_of_mean(zoom_vdm2, f'vdm {gamma2}', (blue_tuple[0], blue_tuple[2]))
    
    plt.xlim([1, zoom_max])
    
    if dataset == 'lego_larger':
        if 'recon' in loss_type:
            plt.ylim([0.005, 0.055])
        elif 'latent' in loss_type:
            plt.ylim([0.0, 0.064])
        elif 'total' in loss_type:
            plt.ylim([1.15, 1.71])
    elif dataset == 'cifar10':
        if 'recon' in loss_type:
            plt.ylim([0.005, 0.085])
        elif 'latent' in loss_type:
            plt.ylim([0.0, 0.04])
        elif 'total' in loss_type:
            plt.ylim([2.9, 3.55])
    else:
        raise ValueError(f'Dataset not implemented: {dataset}')
    
    
    plt.grid()
    plt.legend()
    plt.xlabel('Training step')
    plt.ylabel('Loss in BPD')
    
    if zoom_max > 200000:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    

def get_regex_patterns(
        gamma_ranges: List[str],
        dataset: str,
        model_base_name: str,
        noise_schedule_type: str = 'fixed',
        enc_model: str = 'unet_2',
        edm_extras: str = '',
        batch_size: int = 128,
        diff_size: int = 8) -> List[str]:
        
    g_range_files = []
    for r in gamma_ranges:
        g_range_file = r.replace(' to ', '_').replace('.', '_')
        g_range_files.append(g_range_file)
        
    if edm_extras != '':
        edm_extras = edm_extras + '_'
    
    patterns = []
    for r_f in g_range_files:
        if 'VDM' in model_base_name:
            pattern = f'{model_base_name}_{noise_schedule_type}_{r_f}_unet_{diff_size}_simple_{dataset}_.*_{batch_size}_train'
        else:
            pattern = f'{model_base_name}_{noise_schedule_type}_{r_f}_unet_{diff_size}_{enc_model}_{edm_extras}{dataset}_.*_{batch_size}_train'
        patterns.append(pattern)
    
    return patterns
    

def get_means_with_std_err_and_t_tests_dict(
        folder_path: str, 
        plot_dir: str, 
        dataset: str,
        step: int,
        gamma_range1: str = '-12.8 to 2.0',
        gamma_range2: str = '-13.3 to 5.0') -> Dict:
    
    edm_patterns = get_regex_patterns(
        [gamma_range1, gamma_range2],
        dataset,
        'LE_EPSGAMMA_SIGMAENC')
    
    vdm_patterns = get_regex_patterns(
        [gamma_range1, gamma_range2],
        dataset,
        'VDM2')
    
    
    g_range_file1 = gamma_range1.replace(' to ', '_').replace('.', '_')
    g_range_file2 = gamma_range2.replace(' to ', '_').replace('.', '_')
    
    edm1_pattern = edm_patterns[0]
    edm2_pattern = edm_patterns[1]
    vdm1_pattern = vdm_patterns[0]
    vdm2_pattern = vdm_patterns[1]
    
    combined_edm1 = get_combined_losses(folder_path, edm1_pattern)
    combined_edm2 = get_combined_losses(folder_path, edm2_pattern)
    combined_vdm1 = get_combined_losses(folder_path, vdm1_pattern)
    combined_vdm2 = get_combined_losses(folder_path, vdm2_pattern)
            
    total_test_loss_edm1 = get_total_from_combined_loss_dict(combined_edm1, 'eval')
    if combined_edm2:
        total_test_loss_edm2 = get_total_from_combined_loss_dict(combined_edm2, 'eval')        
    
    total_test_loss_vdm1 = get_total_from_combined_loss_dict(combined_vdm1, 'eval')
    total_test_loss_vdm2 = get_total_from_combined_loss_dict(combined_vdm2, 'eval')
        
    if not combined_edm2:
        keys = [f'edm {g_range_file1}', f'vdm {g_range_file1}', f'vdm {g_range_file2}']
        losses = [total_test_loss_edm1, total_test_loss_vdm1, total_test_loss_vdm2]
        combined_losses = [combined_edm1, combined_vdm1, combined_vdm2]
    else:
        keys = [f'edm {g_range_file1}', f'edm {g_range_file2}', f'vdm {g_range_file1}', f'vdm {g_range_file2}']
        losses = [total_test_loss_edm1, total_test_loss_edm2, total_test_loss_vdm1, total_test_loss_vdm2]
        combined_losses = [combined_edm1, combined_edm2, combined_vdm1, combined_vdm2]
    
   
    
    
    mean_std_err_dict = {}
    
    for i in range(len(keys)):
        current_losses = losses[i]
        losses_step = current_losses[current_losses[:, 0] == float(step)]
        mean, std, std_err = get_loss_mean_std_and_std_err(losses_step)
        
        mean_std_err_dict[f'{keys[i]} total'] = {'mean': mean[0], 'std': std[0], 'std_err': std_err[0]}
    
    loss_names = ['bpd_diff', 'bpd_recon', 'bpd_latent']
    eval_key = 'eval'
    
    for name in loss_names:
        for i in range(len(keys)):
            loss_all_steps = combined_losses[i][name][eval_key]
            loss = loss_all_steps[loss_all_steps[:, 0] == step]
            mean, std, std_err = get_loss_mean_std_and_std_err(loss)
            
            mean_std_err_dict[f'{keys[i]} {name}'] = {'mean': mean[0], 'std': std[0], 'std_err': std_err[0]}
    
    t_test_dict = {}
    edm_vs_vdm_total1 = get_loss_t_test(total_test_loss_edm1, total_test_loss_vdm1, step)
    t_test_dict[f'edm vs vdm {g_range_file1} total'] = {
        'stat': edm_vs_vdm_total1.statistic, 
        'p-value': edm_vs_vdm_total1.pvalue}
    
    
    if not combined_edm2:
        edm_vs_vdm_total2 = get_loss_t_test(total_test_loss_edm1, total_test_loss_vdm2, step)
        
        t_test_dict[f'edm {g_range_file1} vs vdm {g_range_file2} total'] = {
            'stat': edm_vs_vdm_total2.statistic, 
            'p-value': edm_vs_vdm_total2.pvalue}
    else:
        edm_vs_vdm_total2 = get_loss_t_test(total_test_loss_edm2, total_test_loss_vdm2, step)
        
        t_test_dict[f'edm vs vdm {g_range_file2} total'] = {
            'stat': edm_vs_vdm_total2.statistic, 
            'p-value': edm_vs_vdm_total2.pvalue}
    
    if not combined_edm2:
        edm_vs_vdm_recon1 = get_loss_t_test(
            combined_losses[0]['bpd_recon'][eval_key], 
            combined_losses[1]['bpd_recon'][eval_key], 
            step)
        t_test_dict[f'edm vs vdm {g_range_file1} recon'] = {
            'stat': edm_vs_vdm_recon1.statistic, 
            'p-value': edm_vs_vdm_recon1.pvalue}
        
        edm_vs_vdm_latent1 = get_loss_t_test(
            combined_losses[0]['bpd_latent'][eval_key], 
            combined_losses[1]['bpd_latent'][eval_key], 
            step)
        t_test_dict[f'edm vs vdm {g_range_file1} latent'] = {
            'stat': edm_vs_vdm_latent1.statistic, 
            'p-value': edm_vs_vdm_latent1.pvalue}
    else:
        edm_vs_vdm_recon1 = get_loss_t_test(
            combined_losses[0]['bpd_recon'][eval_key], 
            combined_losses[2]['bpd_recon'][eval_key], 
            step)
        t_test_dict[f'edm vs vdm {g_range_file1} recon'] = {
            'stat': edm_vs_vdm_recon1.statistic, 
            'p-value': edm_vs_vdm_recon1.pvalue}
        
        edm_vs_vdm_latent1 = get_loss_t_test(
            combined_losses[0]['bpd_latent'][eval_key], 
            combined_losses[2]['bpd_latent'][eval_key], 
            step)
        t_test_dict[f'edm vs vdm {g_range_file1} latent'] = {
            'stat': edm_vs_vdm_latent1.statistic, 
            'p-value': edm_vs_vdm_latent1.pvalue}
        
        edm_vs_vdm_latent2 = get_loss_t_test(
            combined_losses[1]['bpd_latent'][eval_key], 
            combined_losses[3]['bpd_latent'][eval_key], 
            step)
        t_test_dict[f'edm vs vdm {g_range_file2} latent'] = {
            'stat': edm_vs_vdm_latent2.statistic, 
            'p-value': edm_vs_vdm_latent2.pvalue}
    
    return mean_std_err_dict, t_test_dict


def save_means_with_std_err_and_t_tests_json(
        folder_path: str, 
        plot_dir: str, 
        dataset: str,
        step: int,
        gamma_range1: str = '-12.8 to 2.0',
        gamma_range2: str = '-13.3 to 5.0') -> None:
    
    mean_std_err_dict, t_test_dict = get_means_with_std_err_and_t_tests_dict(
        folder_path, 
        plot_dir,
        dataset,
        step,
        gamma_range1,
        gamma_range2)
    
    g_range_file1 = gamma_range1.replace(' to ', '_').replace('.', '_')
    g_range_file2 = gamma_range2.replace(' to ', '_').replace('.', '_')
    
    mean_std_err_file_name = f'{dataset}_edm_vdm_{g_range_file1}_{g_range_file2}_{step}_mean_std_err.json'
    mean_std_err_file_path = os.path.join(plot_dir, mean_std_err_file_name)
    
    sljson.save_as_json(mean_std_err_dict, mean_std_err_file_path)
    
    t_test_file_name = f'{dataset}_edm_vdm_{g_range_file1}_{g_range_file2}_{step}_t_test.json'
    t_test_file_path = os.path.join(plot_dir, t_test_file_name)
    
    sljson.save_as_json(t_test_dict, t_test_file_path)
    


def get_loss_mean_std_and_std_err(losses):
    std_err = stats.sem(losses[:, 1:], axis = 1)
    std = np.std(losses[:, 1:], axis = 1)
    mean = np.mean(losses[:, 1:], axis = 1)
    
    return mean, std, std_err



def get_loss_t_test(
        all_loss_1: NDArray,
        all_loss_2: NDArray,
        step: int):
    loss_1 = all_loss_1[all_loss_1[:, 0] == step][0, 1:]
    loss_2 = all_loss_2[all_loss_2[:, 0] == step][0, 1:]
    
    result = stats.ttest_ind(loss_1, loss_2, equal_var = False, alternative = 'less')
    
    return result 



def get_means_std_err_and_t_tests_dict(
        folder_path: str, 
        plot_dir: str, 
        dataset: str,
        step: int,
        model_type1: str = 'VDM2_V',
        model_type2: str = 'S3VA',
        noise_schedule_type1: str = 'fixed',
        noise_schedule_type2: str = 'fixed',
        enc_models: Tuple[str, str] = ('unet_2', ''),
        edm_extras: Tuple[str, str] = ('m1_to_1', ''),
        batch_size: int = 128,
        diff_size: int = 8,
        gamma_range1: str = '-13.3 to 5.0',
        gamma_range2: str = '-13.3 to 5.0') -> Dict:
    
    if 'VDM' in model_type1:
        model1_pattern = get_regex_patterns(
            [gamma_range1],
            dataset,
            model_type1,
            noise_schedule_type = noise_schedule_type1,
            batch_size = batch_size,
            diff_size = diff_size)[0]
    else:
        model1_pattern = get_regex_patterns(
            [gamma_range1], 
            dataset,
            model_type1,
            noise_schedule_type = noise_schedule_type1,
            enc_model = enc_models[0],
            edm_extras = edm_extras[0],
            batch_size = batch_size,
            diff_size = diff_size)[0]
    
    if 'VDM' in model_type2:
        model2_pattern = get_regex_patterns(
            [gamma_range2],
            dataset,
            model_type2,
            noise_schedule_type = noise_schedule_type2,
            batch_size = batch_size,
            diff_size = diff_size)[0]
    else:
        model2_pattern = get_regex_patterns(
            [gamma_range1], 
            dataset,
            model_type2,
            noise_schedule_type = noise_schedule_type2,
            enc_model = enc_models[1],
            edm_extras = edm_extras[1],
            batch_size = batch_size,
            diff_size = diff_size)[0]
         
    g_range_file1 = gamma_range1.replace(' to ', '_').replace('.', '_')
    g_range_file2 = gamma_range2.replace(' to ', '_').replace('.', '_')    
    
    combined_1 = get_combined_losses(folder_path, model1_pattern, allow_single_file = True)
    combined_2 = get_combined_losses(folder_path, model2_pattern, allow_single_file = True)
    
    total_test_loss_1 = get_total_from_combined_loss_dict(combined_1, 'eval')
    total_test_loss_2 = get_total_from_combined_loss_dict(combined_2, 'eval')
    
    keys = [f'{model_type1} {noise_schedule_type1} {g_range_file1}', f'{model_type2} {noise_schedule_type2} {g_range_file2}']
    losses = [total_test_loss_1, total_test_loss_2]
    combined_losses = [combined_1, combined_2]
   
    mean_std_err_dict = {}
    
    for i in range(len(keys)):
        current_losses = losses[i]
        losses_step = current_losses[current_losses[:, 0] == float(step)]
        mean, std, std_err = get_loss_mean_std_and_std_err(losses_step)
        
        mean_std_err_dict[f'{keys[i]} total'] = {'mean': mean[0], 'std': std[0], 'std_err': std_err[0]}
    
    loss_names = ['bpd_diff', 'bpd_recon', 'bpd_latent']
    eval_key = 'eval'
    
    for name in loss_names:
        for i in range(len(keys)):
            loss_all_steps = combined_losses[i][name][eval_key]
            loss = loss_all_steps[loss_all_steps[:, 0] == step]
            mean, std, std_err = get_loss_mean_std_and_std_err(loss)
            
            mean_std_err_dict[f'{keys[i]} {name}'] = {'mean': mean[0], 'std': std[0], 'std_err': std_err[0]}
    
    t_test_dict = {}
    model1vs2_total = get_loss_t_test(total_test_loss_1, total_test_loss_2, step)
    t_test_dict[f'{model_type1} {g_range_file1} vs {model_type2} {g_range_file2} total'] = {
        'stat': model1vs2_total.statistic, 
        'p-value': model1vs2_total.pvalue}
    
    model1vs2_latent = get_loss_t_test(
        combined_losses[0]['bpd_latent'][eval_key], 
        combined_losses[1]['bpd_latent'][eval_key], 
        step)
    t_test_dict[f'{model_type1} {g_range_file1} vs {model_type2} {g_range_file2} latent'] = {
        'stat': model1vs2_latent.statistic, 
        'p-value': model1vs2_latent.pvalue}

    model1vs2_recon = get_loss_t_test(
        combined_losses[0]['bpd_recon'][eval_key], 
        combined_losses[1]['bpd_recon'][eval_key], 
        step)
    t_test_dict[f'{model_type1} {g_range_file1} vs {model_type2} {g_range_file2} recon'] = {
        'stat': model1vs2_recon.statistic, 
        'p-value': model1vs2_recon.pvalue}
    
    model1vs2_diff = get_loss_t_test(
        combined_losses[0]['bpd_diff'][eval_key], 
        combined_losses[1]['bpd_diff'][eval_key], 
        step)
    t_test_dict[f'{model_type1} {g_range_file1} vs {model_type2} {g_range_file2} diff'] = {
        'stat': model1vs2_diff.statistic, 
        'p-value': model1vs2_diff.pvalue}
    
    return mean_std_err_dict, t_test_dict












