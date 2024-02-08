#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:32:54 2023

@author: bemi


For inspecting the learned encoder


"""



from flax import linen as nn

import jax
import jax.numpy as jnp
import jax.tree_util 
from jaxtyping import Array

import math
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray

import os

from typing import Dict, List, Tuple

from src.config_classes.ddpm_config import DDPMConfig
from src.config_classes.optimizer_config import OptimizerConfig
from src.config_classes.training_config import TrainingConfig

from src.data import load_data 
from src.evaluation import display

from src.file_handling import naming

from src.model_handling import ModelHandler




class EncoderInspector(ModelHandler):
    
    def __init__(self,
                 train_config: TrainingConfig, 
                 model_config: DDPMConfig,
                 optimizer_config: OptimizerConfig,
                 output_folder: str,
                 date_str: str,
                 step: int):
        
        super().__init__(train_config, model_config, optimizer_config)
        
        self.output_folder = output_folder
        
        self.train_ds, self.eval_ds, self.condition_classes = load_data.load_dataset(
            self.train_config)
        
        prefix = naming.get_file_prefix(self.model_config, self.train_config, self.optimizer_config, date_str)
        restored_state = self.load_checkpoint(output_folder, prefix, step)
        self.train_state = restored_state
        self.prefix = prefix 
        
        self.examples = jnp.zeros((1, 1, 1, 1))
        
            
    
    def get_examples(self, train_eval: str, classes: NDArray, 
                     num_examples: int, 
                     eval_batch_idx: int = 0):
        
        if train_eval == 'train':
            batch = next(self.train_ds)
        else:
            iter_eval = iter(self.eval_ds)
            for i in range(eval_batch_idx + 1):
                batch = next(iter_eval)
        
        images, conditioning = batch
        
        ex_per_class = math.floor(num_examples / len(classes))
        real_num_examples = ex_per_class * len(classes)
        
        examples = np.zeros((real_num_examples, images.shape[1], images.shape[2], images.shape[3]))
        class_ex_so_far = np.zeros(len(classes))
        
        while (class_ex_so_far<ex_per_class).any():
            for i in range(len(classes)):
                filt = conditioning == classes[i]
                examples_so_far = images[filt]
                
                offset = class_ex_so_far[i]
                end = min((i+1)*ex_per_class, i*ex_per_class + len(examples_so_far))
                
                examples[int(i*ex_per_class + offset): end] = images[filt][0:ex_per_class]
                class_ex_so_far[i] = class_ex_so_far[i] + len(examples_so_far)
            
            if (class_ex_so_far<ex_per_class).any():
                if train_eval == 'train':
                    batch = next(self.train_ds)
                else:
                    batch = next(iter(self.eval_ds))
                
                images, conditioning = batch
                
        
        return examples
    
    
    def set_examples(self, 
                     train_eval: str, 
                     num_classes: int, 
                     num_examples: int,
                     eval_batch_idx: int = 0) -> None:
        
        classes = jnp.arange(0, num_classes)
        examples = self.get_examples(train_eval, classes, num_examples, 
                                     eval_batch_idx)
        examples = jnp.array(examples)
        
        self.examples = examples    
    
    
    def get_encoded(self, examples: Array, conditioning: Array, t: float):
        model = self.model 
        state = self.train_state
        
        encoded_examples = state.apply_fn(
            variables = {'params': state.ema_params},
            x = examples, 
            t = t,
            conditioning = conditioning,
            deterministic = True, 
            method = model.encode)
        
        return encoded_examples
    
    
    def get_inner_encoded(self, examples:Array, conditioning: Array, t:float):
        model = self.model 
        state = self.train_state
        
        inner_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples, 
            t = t,
            conditioning = conditioning,
            deterministic = True, 
            method = model.inner_encode)
        
        return inner_encoded_examples
        
     
    def get_encoded_rgb(self, examples: Array, conditioning: Array, t: float):
        model = self.model 
        state = self.train_state
        
        encoded_rgb_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples, 
            t = t,
            conditioning = conditioning,
            deterministic = True, 
            method = model.encode_to_rgb)
        
        return encoded_rgb_examples
    
    def get_sigma_t_squared(self, t: float):
        model = self.model 
        state = self.train_state
        
        gamma_t = state.apply_fn(
            variables= {'params': state.ema_params},
            t = t,
            method = model.get_gamma)
        
        var_t = nn.sigmoid(gamma_t)
        
        return var_t
    
    
    def show_encoded_examples(self, 
                              save_to_folder: str,
                              move_to_0_to_1: bool,
                              scale_up_small: bool,
                              show_late_t: bool,
                              set_new_examples: bool = False) -> None:
        
        examples = self.examples
        cols = math.ceil(len(examples)/ 2)
        
        display.make_image_grid(
            examples.astype(int), 2, cols, self.prefix)
        steps = self.train_state.step
        sample_file_name = f'{self.prefix}sample_{steps}_orig_images.png'
        sample_path = os.path.join(save_to_folder, sample_file_name)
        plt.savefig(sample_path, dpi = 300)
        plt.clf()        
        
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        
        for t_10 in ts:
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.
            
            if move_to_0_to_1:
                encoded_ex = self.get_encoded(examples, np.zeros(len(examples)), t)
                encoded_rgb_ex = display.move_samples_to_0to1_global(encoded_ex, scale_up_small)
                enc_suffix = '0to1'
            else:
                encoded_rgb_ex = self.get_encoded_rgb(examples, np.zeros(len(examples)), t)    
                enc_suffix = 'rgb_int'
                
            display.make_image_grid(
                encoded_rgb_ex, 2, cols, self.prefix)
            
            if save_to_folder == '':
                plt.show()
            else:          
                if show_late_t and t_10 != 0:
                    file_t = t_10 + 90
                else:
                    file_t = t_10
                steps = self.train_state.step
                sample_file_name = f'{self.prefix}sample_{steps}_{enc_suffix}_t_{str(file_t.astype(int))}.png'
                sample_path = os.path.join(save_to_folder, sample_file_name)
                plt.savefig(sample_path, dpi = 300)
                plt.clf()
                
                
    def show_encoded_examples_global_move(self,                               
                              save_to_folder: str,
                              scale_up_small: bool,
                              show_late_t: bool,
                              set_new_examples: bool = False) -> None:
        
        examples = self.examples
        cols = math.ceil(len(examples)/ 2)
        enc_suffix = 'global_0to1'
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        
        t_enc_examples = jnp.zeros((len(ts) + 1, *examples.shape))
        t_enc_examples = t_enc_examples.at[0].set(examples)
        
        for i, t_10 in enumerate(ts):
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.
            encoded_ex = self.get_encoded(examples, np.zeros(len(examples)), t)
            t_enc_examples = t_enc_examples.at[i+1].set(encoded_ex)
                        
        
        t_enc_examples = display.move_samples_to_0to1_global(t_enc_examples, scale_up_small)
        ts = jnp.insert(ts, 0, -1)
        
        for i, t_10 in enumerate(ts):            
            encoded_ex = t_enc_examples[i]
            display.make_image_grid(
                encoded_ex, 2, cols, self.prefix)
            
            if save_to_folder == '':
                plt.show()
            else:                
                if show_late_t and t_10 > 0:
                    file_t = t_10 + 90
                else:
                    file_t = t_10
                steps = self.train_state.step
                sample_file_name = f'{self.prefix}sample_{steps}_{enc_suffix}_t_{str(file_t.astype(int))}.png'
                sample_path = os.path.join(save_to_folder, sample_file_name)
                plt.savefig(sample_path, dpi = 300)
                plt.clf()
                
    
    def visualise_y_part_of_encoding_examples(self, 
                              save_to_folder: str,
                              scale_up_small: bool,
                              show_late_t: bool,
                              move_to_0_to_1: bool = False,
                              move_to_0_to_1_from_m1_1: bool = False,
                              set_new_examples: bool = False) -> None:
        
        examples = self.examples
        
        model = self.model 
        state = self.train_state
        initial_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples,
            method = model.initial_encode)
        
        
        cols = math.ceil(len(examples)/ 2)
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        
        for i, t_10 in enumerate(ts):
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.            
            encoded_ex = self.get_encoded(examples, np.zeros(len(examples)), t)
            inner_encoded_ex = self.get_inner_encoded(examples, np.zeros(len(examples)), t)
            
            enc_vs_orig = encoded_ex - initial_encoded_examples
            
            if move_to_0_to_1:
                inner_encoded_ex_rgb = display.move_samples_to_0to1_global(inner_encoded_ex, scale_up_small)
                enc_suffix = '_0to1'
            elif move_to_0_to_1_from_m1_1:
                inner_encoded_ex_rgb = (inner_encoded_ex + 1)/2
                enc_vs_orig_rgb = (enc_vs_orig + 1)/2
                enc_suffix = '_0to1_from_-1to1'
            else:
                inner_encoded_ex_rgb = inner_encoded_ex
                enc_vs_orig_rgb = enc_vs_orig
                enc_suffix = ''
            
                
            display.make_image_grid(
                enc_vs_orig_rgb, 2, cols, self.prefix)
            
            if save_to_folder == '':
                plt.show()
                plt.close()
            else:     
                if show_late_t and t_10 > 0:
                    file_t = t_10 + 90
                else:
                    file_t = t_10
                steps = self.train_state.step
                sample_file_name = f'{self.prefix}sample_{steps}_enc-orig{enc_suffix}_t_{str(file_t.astype(int))}.png'
                sample_path = os.path.join(save_to_folder, sample_file_name)
                plt.savefig(sample_path, dpi = 300)
                plt.clf()
                plt.close()
            
            display.make_image_grid(
                inner_encoded_ex_rgb, 2, cols, self.prefix)
            
            if save_to_folder == '':
                plt.show()
                plt.close()
            else:     
                if show_late_t and t_10 > 0:
                    file_t = t_10 + 90
                else:
                    file_t = t_10
                steps = self.train_state.step
                sample_file_name = f'{self.prefix}sample_{steps}_innerenc{enc_suffix}_t_{str(file_t.astype(int))}.png'
                sample_path = os.path.join(save_to_folder, sample_file_name)
                plt.savefig(sample_path, dpi = 300)
                plt.clf()
                plt.close()
                
                
    def get_amplified_encoding(self, 
                              save_to_folder: str,
                              scale_up_small: bool,
                              show_late_t: bool,
                              amplifier: float,
                              move_to_0_to_1: bool = False,
                              move_to_0_to_1_from_m1_1: bool = False,
                              set_new_examples: bool = False) -> None:
        
        examples = self.examples
        
        model = self.model 
        state = self.train_state
        initial_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples,
            method = model.initial_encode)
        
        
        cols = math.ceil(len(examples)/ 2)
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        
        for i, t_10 in enumerate(ts):
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.            
            encoded_ex = self.get_encoded(examples, np.zeros(len(examples)), t)
            
            enc_vs_orig = encoded_ex - initial_encoded_examples
            amplified_diff = amplifier * enc_vs_orig
            
            amplified_encoded = amplified_diff + initial_encoded_examples
            
            if move_to_0_to_1:                
                amplified_encoded = display.move_samples_to_0to1_global(amplified_encoded, scale_up_small)
                enc_suffix = '_0to1'
            elif move_to_0_to_1_from_m1_1:
                amplified_encoded = (amplified_encoded + 1)/2
                enc_suffix = '_0to1_from_-1to1'
            else:                
                enc_suffix = ''
            
                
            display.make_image_grid(
                amplified_encoded, 2, cols, self.prefix)
            
            if save_to_folder == '':
                plt.show()
                plt.close()
            else:     
                if show_late_t and t_10 > 0:
                    file_t = t_10 + 90
                else:
                    file_t = t_10
                steps = self.train_state.step
                amp_str = str(amplifier).replace('.', '_')
                sample_file_name = f'{self.prefix}sample_{steps}_amp{amp_str}_enc{enc_suffix}_t_{str(file_t.astype(int))}.png'
                sample_path = os.path.join(save_to_folder, sample_file_name)
                plt.savefig(sample_path, dpi = 300)
                plt.clf()
                plt.close()
            
    
                
    def visualise_orig_enc_diff_scaled_up_by_sigma(self, 
                              save_to_folder: str,
                              show_late_t: bool,                
                              move_to_0_to_1_from_m2_2: bool = False,
                              set_new_examples: bool = False) -> None:
        
        examples = self.examples
        
        model = self.model 
        state = self.train_state
        initial_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples,
            method = model.initial_encode)
        
        
        cols = math.ceil(len(examples)/ 2)
        
        display.make_image_grid(
            examples.astype(int), 2, cols, self.prefix)
        steps = self.train_state.step
        sample_file_name = f'{self.prefix}sample_{steps}_inner-orig-diff_orig_images.png'
        sample_path = os.path.join(save_to_folder, sample_file_name)
        plt.savefig(sample_path, dpi = 300)
        plt.clf()        
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        
        for i, t_10 in enumerate(ts):
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.            
            # x_t - x = x - sigma_t**2 x + sigma_t**2 y - x = sigma_t**2 (y - x)
            # Visualise (y - x) where y is the inner encoder
            
            inner_encoded_ex = self.get_inner_encoded(examples, np.zeros(len(examples)), t)
            
            inner_orig_diff = inner_encoded_ex - initial_encoded_examples
            
            if move_to_0_to_1_from_m2_2:
                print(f'Before min: {np.min(inner_orig_diff)}, before max: {np.max(inner_orig_diff)}')
                inner_orig_diff = (inner_orig_diff + 2)/4
                print(f'After min: {np.min(inner_orig_diff)}, after max: {np.max(inner_orig_diff)}')
                enc_suffix = '_0to1_from_-2to2'
            else:                
                enc_suffix = ''
            
                
            display.make_image_grid(
                inner_orig_diff, 2, cols, self.prefix)
            
            if save_to_folder == '':
                plt.show()
                plt.close()
            else:     
                if show_late_t and t_10 > 0:
                    file_t = t_10 + 90
                else:
                    file_t = t_10
                steps = self.train_state.step
                sample_file_name = f'{self.prefix}sample_{steps}_inner-orig-diff{enc_suffix}_t_{str(file_t.astype(int))}.png'
                sample_path = os.path.join(save_to_folder, sample_file_name)
                plt.savefig(sample_path, dpi = 300)
                plt.clf()
                plt.close()
                
                
    def orig_enc_diff_scaled_up_by_sigma_heatmaps(self, 
                              save_to_folder: str,
                              show_late_t: bool,                
                              move_to_0_to_1_from_m2_2: bool = False) -> None:
        
        examples = self.examples
        
        model = self.model 
        state = self.train_state
        initial_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples,
            method = model.initial_encode)
        
        
        cols = math.ceil(len(examples)/ 2)
        
        int_examples = examples.astype(int)
        display.make_image_grid(
            int_examples, 2, cols, self.prefix)
        steps = self.train_state.step
        sample_file_name = f'{self.prefix}sample_{steps}_sum_heatmap_orig_images.png'
        sample_path = os.path.join(save_to_folder, sample_file_name)
        plt.savefig(sample_path, dpi = 300)
        plt.clf()
        
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        
        # Init for setting inner_encoded_ex_s for steps i > 0 
        # and i < (len(ts) - 1) 
        encoded_ex_t = initial_encoded_examples
        
        for i, t_10 in enumerate(ts):
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.            
            
            if i > 0:
                encoded_ex_s = encoded_ex_t  
            
            encoded_ex_t = self.get_encoded(examples, np.zeros(len(examples)), t)
            
            if i > 0:
                encoded_ex_t_s_diff = encoded_ex_t - encoded_ex_s
                if show_late_t and t_10 != 0:
                    s = (t_10-1+90)/100.
                else:
                    s = (t_10-1)/10.     
                tau = t - s
                
                encoded_ex_t_s_diff = encoded_ex_t_s_diff/tau
                encoded_ex_t_s_diff_sum = jnp.sum(encoded_ex_t_s_diff, axis = 3)
                
                display.make_image_grid(encoded_ex_t_s_diff_sum, 2, cols, self.prefix, cmap = 'bwr')
                        
                if save_to_folder == '':
                    plt.show()
                    plt.close()
                else:     
                    if show_late_t and t_10 > 0:
                        file_t = t_10 + 90
                    else:
                        file_t = t_10
                    steps = self.train_state.step
                    sample_file_name = f'{self.prefix}sample_{steps}_sum_heatmap_x_t_s_diff_t_{str(file_t.astype(int))}.png'
                    sample_path = os.path.join(save_to_folder, sample_file_name)
                    plt.savefig(sample_path, dpi = 300)
                    plt.clf()
                    plt.close()
                
                
                encoded_ex_s_t_diff = encoded_ex_s - encoded_ex_t
                                
                encoded_ex_s_t_diff = encoded_ex_s_t_diff/tau
                encoded_ex_s_t_diff_sum = jnp.sum(encoded_ex_s_t_diff, axis = 3)
                
                display.make_image_grid(encoded_ex_s_t_diff_sum, 2, cols, self.prefix, cmap = 'bwr')
                        
                if save_to_folder == '':
                    plt.show()
                    plt.close()
                else:     
                    if show_late_t and t_10 > 0:
                        file_t = t_10 + 90
                    else:
                        file_t = t_10
                    steps = self.train_state.step
                    sample_file_name = f'{self.prefix}sample_{steps}_sum_heatmap_x_s_t_diff_t_{str(file_t.astype(int))}.png'
                    sample_path = os.path.join(save_to_folder, sample_file_name)
                    plt.savefig(sample_path, dpi = 300)
                    plt.clf()
                    plt.close()
                
                
                # # max min diff
                # encoded_ex_t_s_diff_max = jnp.max(encoded_ex_t_s_diff, axis = 3)
                # encoded_ex_t_s_diff_min = jnp.min(encoded_ex_t_s_diff, axis = 3)
                # encoded_ex_t_s_diff_max_min = encoded_ex_t_s_diff_max - encoded_ex_t_s_diff_min
                
                # display.make_image_grid(encoded_ex_t_s_diff_max_min, 2, cols, self.prefix, cmap = 'inferno')
                        
                # if save_to_folder == '':
                #     plt.show()
                #     plt.close()
                # else:     
                #     if show_late_t and t_10 > 0:
                #         file_t = t_10 + 90
                #     else:
                #         file_t = t_10
                #     steps = self.train_state.step
                #     sample_file_name = f'{self.prefix}sample_{steps}_max_m_min_heatmap_x_t_s_diff_t_{str(file_t.astype(int))}.png'
                #     sample_path = os.path.join(save_to_folder, sample_file_name)
                #     plt.savefig(sample_path, dpi = 300)
                #     plt.clf()
                #     plt.close()
                
                
                # std 
                encoded_ex_t_s_diff_std = jnp.std(encoded_ex_t_s_diff, axis = 3)
                
                display.make_image_grid(encoded_ex_t_s_diff_std, 2, cols, self.prefix, cmap = 'inferno')
                        
                if save_to_folder == '':
                    plt.show()
                    plt.close()
                else:     
                    if show_late_t and t_10 > 0:
                        file_t = t_10 + 90
                    else:
                        file_t = t_10
                    steps = self.train_state.step
                    sample_file_name = f'{self.prefix}sample_{steps}_std_channel_heatmap_x_t_s_diff_t_{str(file_t.astype(int))}.png'
                    sample_path = os.path.join(save_to_folder, sample_file_name)
                    plt.savefig(sample_path, dpi = 300)
                    plt.clf()
                    plt.close()
                
                
            
            # x_t - x = x - sigma_t**2 x + sigma_t**2 y - x = sigma_t**2 (y - x)
            # Visualise (y - x) where y is the inner encoder
            inner_encoded_ex = self.get_inner_encoded(examples, np.zeros(len(examples)), t)
            
            inner_orig_diff = inner_encoded_ex - initial_encoded_examples
            inner_orig_diff_sum = jnp.sum(inner_orig_diff, axis = 3)
            print(f'Sum min: {np.min(inner_orig_diff_sum)}, sum max: {np.max(inner_orig_diff_sum)}')
            
            
            if move_to_0_to_1_from_m2_2:
                print(f'Before min: {np.min(inner_orig_diff)}, before max: {np.max(inner_orig_diff)}')
                inner_orig_diff_sum = (inner_orig_diff_sum + 2)/4
                inner_orig_diff = (inner_orig_diff + 2)/4
                print(f'After min: {np.min(inner_orig_diff)}, after max: {np.max(inner_orig_diff)}')
                enc_suffix = '_0to1_from_-2to2'
            else:                
                enc_suffix = ''
            
            display.make_image_grid(inner_orig_diff_sum, 2, cols, self.prefix, cmap = 'bwr')
                    
            if save_to_folder == '':
                plt.show()
                plt.close()
            else:     
                if show_late_t and t_10 > 0:
                    file_t = t_10 + 90
                else:
                    file_t = t_10
                steps = self.train_state.step
                sample_file_name = f'{self.prefix}sample_{steps}_sum_heatmap{enc_suffix}_t_{str(file_t.astype(int))}.png'
                sample_path = os.path.join(save_to_folder, sample_file_name)
                plt.savefig(sample_path, dpi = 300)
                plt.clf()
                plt.close()
            
            # for channel in range(3):
            #     channel_im = inner_orig_diff[:, :, :, channel]
            #     display.make_image_grid(channel_im, 2, cols, self.prefix, cmap = 'bwr')
                        
            #     if save_to_folder == '':
            #         plt.show()
            #         plt.close()
            #     else:     
            #         if show_late_t and t_10 > 0:
            #             file_t = t_10 + 90
            #         else:
            #             file_t = t_10
            #         steps = self.train_state.step
            #         sample_file_name = f'{self.prefix}sample_{steps}_t_{str(file_t.astype(int))}_heatmap{enc_suffix}_{channel}.png'
            #         sample_path = os.path.join(save_to_folder, sample_file_name)
            #         plt.savefig(sample_path, dpi = 300)
            #         plt.clf()
            #         plt.close()
                
            #     channel_im = inner_orig_diff
            #     for i in range(3):
            #         if i != channel:
            #             channel_im = channel_im.at[:, :, :, i].set(0)
                
            #     display.make_image_grid(channel_im, 2, cols, self.prefix, cmap = 'bwr')
                        
            #     if save_to_folder == '':
            #         plt.show()
            #         plt.close()
            #     else:     
            #         if show_late_t and t_10 > 0:
            #             file_t = t_10 + 90
            #         else:
            #             file_t = t_10
            #         steps = self.train_state.step
            #         sample_file_name = f'{self.prefix}sample_{steps}_t_{str(file_t.astype(int))}_heatmap_channel{enc_suffix}_{channel}.png'
            #         sample_path = os.path.join(save_to_folder, sample_file_name)
            #         plt.savefig(sample_path, dpi = 300)
            #         plt.clf()
            #         plt.close()
        
    
    def get_article_heatmaps_x_t_changes(self, 
                              save_to_folder: str,
                              show_late_t: bool,
                              ts_for_table: List[int] = [5, 6, 8, 9],
                              example_idxs: List[int] = [1, 3, 4, 7],
                              move_to_0_to_1_from_m2_2: bool = False) -> None:
        
        examples = self.examples
        examples = examples[jnp.array(example_idxs)]
        cols = len(ts_for_table) + 1
        rows = len(example_idxs)
        
        model = self.model 
        state = self.train_state
        initial_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples,
            method = model.initial_encode)
        
        im_list = [0 for i in range(len(examples) * cols)]
        
        initial_encoded_examples = (initial_encoded_examples + 1)/2
        for i in range(rows):
            im_list[i*cols] = initial_encoded_examples[i]
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        
        # Init for setting inner_encoded_ex_s for steps i > 0 
        # and i < (len(ts) - 1) 
        encoded_ex_t = initial_encoded_examples
        col = 1
        for i, t_10 in enumerate(ts):
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.            
            
            if i > 0:
                encoded_ex_s = encoded_ex_t  
            
            encoded_ex_t = self.get_encoded(examples, np.zeros(len(examples)), t)
            
            if t_10 in ts_for_table:
                if i > 0:
                    encoded_ex_t_s_diff = encoded_ex_t - encoded_ex_s
                    if show_late_t and t_10 != 0:
                        s = (t_10-1+90)/100.
                    else:
                        s = (t_10-1)/10.     
                    tau = t - s
                    
                    encoded_ex_t_s_diff = encoded_ex_t_s_diff/tau
                    encoded_ex_t_s_diff_sum = jnp.sum(encoded_ex_t_s_diff, axis = 3)
                    
                    for j in range(rows):
                        im_list[j*cols + col] = encoded_ex_t_s_diff_sum[j]
                        
                    col +=1                
        
        
        fontsize = 'medium' if cols < 10 else 'xx-small'  
        if cols < 6:
            fontsize = 'x-large'
        
        t_cols = [f'0.{n}' for n in ts_for_table if n != 10]
        if 10 in ts_for_table:
            t_cols = t_cols + ['1.0']
        col_labels = ['original'] + t_cols
        
        display.make_image_grid_special_first_col(
            im_list, rows, cols, self.prefix, col_labels = col_labels, 
            cmap = 'bwr', fontsize = fontsize)
                
        if save_to_folder == '':
            plt.show()
            plt.close()
        else:     
            file_exs = '_'.join([str(n) for n in example_idxs])
            file_t = '_'.join([str(n) for n in ts_for_table])
            steps = self.train_state.step
            sample_file_name = f'{self.prefix}sample_{steps}_orig_heatmap_x_t_s_diff_ts_{file_t}_ex_{file_exs}.png'
            sample_path = os.path.join(save_to_folder, sample_file_name)
            plt.savefig(sample_path, dpi = 300, bbox_inches='tight')
            plt.clf()
            plt.close()
            
            
    def get_article_heatmaps_orig_x_t_diff(self, 
                              save_to_folder: str,
                              show_late_t: bool,
                              ts_for_table: List[int] = [5, 6, 7, 8, 9],
                              example_idxs: List[int] = [1, 3, 4, 7],
                              move_to_0_to_1_from_m2_2: bool = False) -> None:
        
        examples = self.examples
        examples = examples[jnp.array(example_idxs)]
        cols = len(ts_for_table) + 1
        rows = len(example_idxs)
        
        model = self.model 
        state = self.train_state
        initial_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples,
            method = model.initial_encode)
        
        im_list = [0 for i in range(len(examples) * cols)]
        
        initial_encoded_examples = (initial_encoded_examples + 1)/2
        for i in range(rows):
            im_list[i*cols] = initial_encoded_examples[i]
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        
        # x_t - x = x - sigma_t**2 x + sigma_t**2 y - x = sigma_t**2 (y - x)
        # Visualise (y - x) where y is the inner encoder
        col = 1
        for i, t_10 in enumerate(ts):
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.            
            
            if t_10 in ts_for_table:
                inner_encoded_ex = self.get_inner_encoded(examples, np.zeros(len(examples)), t)
                inner_orig_diff = inner_encoded_ex - initial_encoded_examples
                
                inner_orig_diff_sum = jnp.sum(inner_orig_diff, axis = 3)
                
                if move_to_0_to_1_from_m2_2:
                    print(f'Before min: {np.min(inner_orig_diff)}, before max: {np.max(inner_orig_diff)}')
                    inner_orig_diff_sum = (inner_orig_diff_sum + 1)/2
                                
                for j in range(rows):
                    im_list[j*cols + col] = inner_orig_diff_sum[j]
                    
                col +=1       
            
        
        if move_to_0_to_1_from_m2_2:            
            enc_suffix = '_0to1_from_-2to2'
        else:                
            enc_suffix = ''
        
        fontsize = 'medium' if cols < 8 else 'xx-small'            
        
        t_cols = [f'0.{n}' for n in ts_for_table if n != 10]
        if 10 in ts_for_table:
            t_cols = t_cols + ['1.0']
        col_labels = ['original'] + t_cols
        
        display.make_image_grid_special_first_col(
            im_list, rows, cols, self.prefix, col_labels = col_labels, 
            cmap = 'bwr', fontsize = fontsize)
        
        if save_to_folder == '':
            plt.show()
            plt.close()
        else:     
            if show_late_t and t_10 > 0:
                file_t = t_10 + 90
            else:
                file_t = t_10
            file_exs = '_'.join([str(n) for n in example_idxs])
            file_t = '_'.join([str(n) for n in ts_for_table])
            steps = self.train_state.step
            sample_file_name = f'{self.prefix}sample_{steps}_sum_heatmap{enc_suffix}_ts_{file_t}_ex_{file_exs}.png'
            sample_path = os.path.join(save_to_folder, sample_file_name)
            plt.savefig(sample_path, dpi = 300)
            plt.clf()
            plt.close()
            
    
    def get_encoded_for_ts(self, 
                              save_to_folder: str,
                              ts_for_table: List[float] = [0.0, 0.70, 0.80, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0],
                              example_idxs: List[int] = [0, 3, 6, 9],
                              move_to_0_to_1_from_m1_1: bool = False) -> None:
        
        examples = self.examples
        examples = examples[jnp.array(example_idxs)]
        cols = len(ts_for_table)
        rows = len(example_idxs)
        
        model = self.model 
        state = self.train_state
        initial_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples,
            method = model.initial_encode)
        
        im_list = [0 for i in range(len(examples) * cols)]
                        
        # Init for setting inner_encoded_ex_s for steps i > 0 
        # and i < (len(ts) - 1) 
        encoded_ex_t = initial_encoded_examples
        col = 0
        for i, t in enumerate(ts_for_table):
            
            encoded_ex_t = self.get_encoded(examples, np.zeros(len(examples)), t)
            
            if move_to_0_to_1_from_m1_1:
                encoded_ex_t = (encoded_ex_t + 1)/2
            
            for j in range(rows):
                im_list[j*cols + col] = encoded_ex_t[j]
            col += 1
                
                
        fontsize = 'medium' if cols < 10 else 'xx-small'            
        
        col_labels = ['{0:.2f}'.format(n) for n in ts_for_table]
        
        display.make_image_grid(im_list, rows, cols, self.prefix, 
                                col_labels = col_labels, fontsize = fontsize,
                                axes_pad = 0.05, text_x_y = (2.5, -15.0))
                
        if save_to_folder == '':
            plt.show()
            plt.close()
        else:     
            file_exs = '_'.join([str(n) for n in example_idxs])
            file_t = '-'.join([str(n).replace('.', '_') for n in ts_for_table])
            steps = self.train_state.step
            sample_file_name = f'{self.prefix}sample_{steps}_encoded_ts_{file_t}_ex_{file_exs}.png'
            sample_path = os.path.join(save_to_folder, sample_file_name)
            plt.savefig(sample_path, dpi = 300, bbox_inches='tight')
            plt.clf()
            plt.close()
    
        

    def plot_encoded_values_vs_t(self, 
                              train_eval: str, 
                              num_classes: int, 
                              num_examples: int,
                              save_to_folder: str) -> None:
            
        model_name = naming.get_model_name(self.model_config, self.train_config, self.optimizer_config)
        model = self.model 
        state = self.train_state
        step = state.step
        
        examples = self.examples
        
        ex_len = examples.shape[0]
        
        ts = jnp.arange(0, 1.01, step= 0.01)
        
        ts_rep = jnp.repeat(ts, ex_len)
        examples_rep = jnp.vstack(jnp.repeat(examples.reshape(1, *examples.shape), len(ts), axis = 0))
        model = self.model 
        state = self.train_state
        step = state.step
        
        encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples_rep, 
            t = ts_rep,
            conditioning = np.zeros(len(ts_rep)),
            deterministic = True, 
            method = model.encode)
        
        with plt.rc_context({'axes.prop_cycle': plt.cycler(color=['b', 'orange', 'green'])}):
            for j in range(ex_len):
                current_ex = encoded_examples[j::ex_len]
                
                min_enc = jnp.min(current_ex, axis = [1, 2, 3]).reshape(-1,1)
                mean_enc = jnp.mean(current_ex, axis = [1, 2, 3]).reshape(-1, 1)
                max_enc = jnp.max(current_ex, axis = [1, 2, 3]).reshape(-1, 1)
                
                min_mean_max = np.hstack([min_enc, mean_enc, max_enc])
                
                plt.plot(ts, min_mean_max, linewidth=1)        
        
        
        plt.title(f'x_t min-mean-max-values vs t - {step}')
        
        if save_to_folder == '':
            plt.show()
        else:        
            plot_file_name = f'{self.prefix}{step}_x_t_min_mean_max_vs_t.png'
            sample_path = os.path.join(save_to_folder, plot_file_name)
            plt.savefig(sample_path, dpi = 300)
        
        plt.clf()
        # Alpha times x_t
        
        
        g_ts = state.apply_fn(
            variables= {'params': state.ema_params},
            t = ts, 
            method = model.get_gamma)
        alpha_ts = jnp.sqrt(nn.sigmoid(-g_ts)).reshape(-1, 1, 1, 1)
        
        with plt.rc_context({'axes.prop_cycle': plt.cycler(color=['b', 'orange', 'green'])}):
            for j in range(ex_len):
                current_ex = encoded_examples[j::ex_len]
                current_ex = alpha_ts * current_ex
                
                min_enc = jnp.min(current_ex, axis = [1, 2, 3]).reshape(-1,1)
                mean_enc = jnp.mean(current_ex, axis = [1, 2, 3]).reshape(-1, 1)
                max_enc = jnp.max(current_ex, axis = [1, 2, 3]).reshape(-1, 1)
                
                min_mean_max = np.hstack([min_enc, mean_enc, max_enc])
                
                plt.plot(ts, min_mean_max, linewidth=1)
        
                
        plt.title(f'alpha * x_t min-mean-max-values vs t - {step}')
        
        if save_to_folder == '':
            plt.show()
        else:        
            plot_file_name = f'{self.prefix}{step}_alpha_x_t_min_mean_max_vs_t.png'
            sample_path = os.path.join(save_to_folder, plot_file_name)
            plt.savefig(sample_path, dpi = 300)
            plt.clf()
            
            
    def plot_inner_encoded_values_vs_t(self, 
                              train_eval: str, 
                              num_classes: int, 
                              num_examples: int,
                              save_to_folder: str) -> None:
        
        model_name = naming.get_model_name(self.model_config, self.train_config, self.optimizer_config)
        if 'ENCPARAM' not in model_name and 'SIGMAENC' not in model_name:
            raise ValueError('Inner encoder only defined for encoders with a parameterized encoder.')
       
        examples = self.examples
        
        ex_len = examples.shape[0]
        
        ts = jnp.arange(0, 1.01, step= 0.01)
        
        ts_rep = jnp.repeat(ts, ex_len)
        examples_rep = jnp.vstack(jnp.repeat(examples.reshape(1, *examples.shape), len(ts), axis = 0))
        model = self.model 
        state = self.train_state
        step = state.step
        
        inner_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples_rep, 
            t = ts_rep,
            conditioning = np.zeros(len(ts_rep)),
            deterministic = True, 
            method = model.inner_encode)
        
        
        for j in range(ex_len):
            current_ex = inner_encoded_examples[j::ex_len]
            current_ex_norm = jnp.sqrt(jnp.sum(jnp.square(current_ex), axis = (1, 2, 3)))
            plt.plot(ts, current_ex_norm, linewidth=1, c = 'b')
    
        plt.title(f'y_x_t norm vs t - {step}')
        
        if save_to_folder == '':
            plt.show()
        else:        
            plot_file_name = f'{self.prefix}{step}_norm_y_t_vs_t.png'
            sample_path = os.path.join(save_to_folder, plot_file_name)
            plt.savefig(sample_path, dpi = 300)
        
        plt.clf()
        
        g_ts = state.apply_fn(
            variables= {'params': state.ema_params},
            t = ts, 
            method = model.get_gamma)
        sigma_ts = jnp.sqrt(nn.sigmoid(g_ts)).reshape(-1, 1, 1, 1)
        
        for j in range(ex_len):
            current_ex = inner_encoded_examples[j::ex_len]
            current_ex = sigma_ts*current_ex
            current_ex_norm = jnp.sqrt(jnp.sum(jnp.square(current_ex), axis = (1, 2, 3)))
            plt.plot(ts, current_ex_norm, linewidth=1, c = 'b')
    
        plt.title(f'sigma_t*y_x_t norm vs t - {step}')
        
        if save_to_folder == '':
            plt.show()
        else:        
            plot_file_name = f'{self.prefix}{step}_norm_sigma_y_t_vs_t.png'
            sample_path = os.path.join(save_to_folder, plot_file_name)
            plt.savefig(sample_path, dpi = 300)
        
        plt.clf()
        
        alpha_ts = jnp.sqrt(nn.sigmoid(-g_ts)).reshape(-1, 1, 1, 1)
        
        initial_encoded_examples = state.apply_fn(
            variables= {'params': state.ema_params},
            x = examples,
            method = model.initial_encode)
        initial_encoded_rep = jnp.vstack(jnp.repeat(initial_encoded_examples.reshape(1, *initial_encoded_examples.shape), len(ts), axis = 0))
        
        for j in range(ex_len):
            current_ex = initial_encoded_rep[j::ex_len]
            current_ex = alpha_ts*current_ex
            current_ex_norm = jnp.sqrt(jnp.sum(jnp.square(current_ex), axis = (1, 2, 3)))
            plt.plot(ts, current_ex_norm, linewidth=1, c = 'b')
    
        plt.title(f'alpha_t*x norm vs t - {step}')
        
        if save_to_folder == '':
            plt.show()
        else:        
            plot_file_name = f'{self.prefix}{step}_norm_alpha_t_x_vs_t.png'
            sample_path = os.path.join(save_to_folder, plot_file_name)
            plt.savefig(sample_path, dpi = 300)
    
        
        plt.clf()
        
        plt.yscale('log')
        
        for j in range(ex_len):
            alpha_ex = alpha_ts*initial_encoded_rep[j::ex_len]            
            alpha_ex_norm = jnp.sqrt(jnp.sum(jnp.square(alpha_ex), axis = (1, 2, 3)))
            if j==0:
                plt.plot(ts, alpha_ex_norm, linewidth=1, c = 'b', label = 'alpha*x')
            else:
                plt.plot(ts, alpha_ex_norm, linewidth=1, c = 'b')
                
            
            
            sigma_y_t = sigma_ts*inner_encoded_examples[j::ex_len]
            sigma_y_t_norm = jnp.sqrt(jnp.sum(jnp.square(sigma_y_t), axis = (1, 2, 3)))
            if j==0:
                plt.plot(ts, sigma_y_t_norm, linewidth=1, c = 'r', label = 'sigma*y_t')
            else:
                plt.plot(ts, sigma_y_t_norm, linewidth=1, c = 'r')
                
    
        plt.title(f'alpha_t*x and sigma_t*y_x_t norm vs t - {step}')
        plt.legend()
        
        if save_to_folder == '':
            plt.show()
        else:        
            plot_file_name = f'{self.prefix}{step}_norm_alpha_t_x_and_sigma_y_t_vs_t.png'
            sample_path = os.path.join(save_to_folder, plot_file_name)
            plt.savefig(sample_path, dpi = 300)
            plt.clf()
            
    
    def get_statistics(self, 
                              train_eval: str, 
                              seed: int,                              
                              show_late_t: bool) -> Dict:
        
        if train_eval == 'train':
            batch = next(self.train_ds)
        else:
            batch = next(iter(self.eval_ds))
        
        images, conditioning = batch
        images = jax.tree_util.tree_map(jnp.asarray, images)
        conditioning = jax.tree_util.tree_map(jnp.asarray, conditioning)
        
        
        model = self.model 
        state = self.train_state
        init_x = state.apply_fn(
            variables= {'params': state.ema_params},
            x = images,
            method = model.initial_encode)
        
        stats_dict = {}
        init_var = np.var(init_x).item()
        stats_dict['original'] = {
            'mean': np.mean(init_x).item(), 
            'variance': init_var}
        
        stats_dict['alpha_original+noise'] = {}
        stats_dict['original+enc'] = {}
        stats_dict['alpha_(original+enc)+noise'] = {}
        stats_dict['pred_stats'] = {}
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        rng = np.random.default_rng(seed=seed)
        
        for t_10 in ts:
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.
            
            encoded_ex = self.get_encoded(images, conditioning, t)
            
            t_key = str(int(t_10))
            
            stats_dict['original+enc'][t_key] = {
                'mean': np.mean(encoded_ex).item(), 
                'variance': np.var(encoded_ex).item()}
            
            eps = rng.normal(size = encoded_ex.shape)
            
            var_t = self.get_sigma_t_squared(t)
            alpha_t = jnp.sqrt(1-var_t)
            
            alpha_original = alpha_t*init_x
            alpha_original_noise = alpha_original + jnp.sqrt(var_t)*eps
            
            stats_dict['alpha_original+noise'][t_key] = {
                'mean': np.mean(alpha_original_noise).item(), 
                'variance': np.var(alpha_original_noise).item()
                }
            
            
            alpha_enc_noise = alpha_t*encoded_ex + jnp.sqrt(var_t)*eps
            
            stats_dict['alpha_(original+enc)+noise'][t_key] = {
                'mean': np.mean(alpha_enc_noise).item(), 
                'variance': np.var(alpha_enc_noise).item()
                }
            
            
            pred_var = (1-var_t)*init_var + var_t
            stats_dict['pred_stats'][t_key] = {
                'mean': 'NotImplemented', 
                'variance': pred_var.item()
                }
        
        return stats_dict    
    
    def get_inner_encoded_and_gradients(self,
                              ts: List[float],                              
                              set_new_examples: bool = False) -> Tuple[List, List]:
        
        examples = self.examples
                
        inner_encoded_at_ts = []
        inner_encoded_approx_gradients = []
        
        for i, t in enumerate(ts):
            inner_encoded_ex = self.get_inner_encoded(examples, np.zeros(len(examples)), t)
            
            inner_encoded_at_ts.append(inner_encoded_ex)
            
            if i > 0:
                previous_t = ts[i-1]
                change_from_prev_t = (inner_encoded_ex - inner_encoded_at_ts[i-1])/(t-previous_t)
                inner_encoded_approx_gradients.append(change_from_prev_t)
            
        return inner_encoded_at_ts, inner_encoded_approx_gradients
        
        
    
    def get_workshop_examples(self, 
                              train_eval: str, 
                              num_classes: int, 
                              num_examples: int,
                              save_to_folder: str,
                              scale_up_small: bool = False,
                              show_late_t: bool = True,
                              set_new_examples: bool = False) -> None:
        
        if set_new_examples:
            self.set_examples(train_eval, num_classes, num_examples)
                
        examples = self.examples
        enc_suffix = 'global_0to1'
        
        ts = jnp.arange(0, 11, step= 1, dtype = jnp.float32)
        
        t_enc_examples = jnp.zeros((len(ts) + 1, *examples.shape))
        t_enc_examples = t_enc_examples.at[0].set(examples)
        
        for i, t_10 in enumerate(ts):
            if show_late_t and t_10 != 0:
                t = (t_10+90)/100.
            else:
                t = t_10/10.
            encoded_ex = self.get_encoded(examples, np.zeros(len(examples)), t)
            t_enc_examples = t_enc_examples.at[i+1].set(encoded_ex)
                        
        
        t_enc_examples = display.move_samples_to_0to1_global(t_enc_examples, scale_up_small)
        
        example_t_indexes = jnp.array([0,6,9,10,11])
        example_brick_indexes = jnp.array([6, 0, 3, 7, 8])
        ncols = len(example_t_indexes)
        nrows = len(example_brick_indexes)
        
        ex_to_plot = t_enc_examples[example_t_indexes][:, example_brick_indexes]
        
        ex_to_plot_ordered = jnp.zeros((nrows*ncols, *examples.shape[1:]))
        
        col_indexes = jnp.arange(nrows)*ncols        
        for i in range(ncols):
            ex_to_plot_ordered = ex_to_plot_ordered.at[col_indexes + i].set(ex_to_plot[i])
        
        t_labels = [f't = {(t_10+90)/100.:0.2f}' 
                    for t_10 in example_t_indexes - 1
                    if t_10 >= 0]
        all_labels = ['original'] + t_labels
        
        display.make_image_grid(ex_to_plot_ordered, 
                                nrows, ncols, 
                                self.prefix,
                                all_labels)
        
        if save_to_folder == '':
            plt.show()
        else:                
            steps = self.train_state.step
            sample_file_name = f'{self.prefix}sample_{steps}_enc_{nrows}_{ncols}_{enc_suffix}.png'
            sample_path = os.path.join(save_to_folder, sample_file_name)
            plt.savefig(sample_path, dpi = 300, bbox_inches='tight')
            plt.clf()     
    
    
    
    
    
    








