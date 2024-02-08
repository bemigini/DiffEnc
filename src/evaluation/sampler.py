#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:02:45 2023

@author: bemi


For drawing samples from a model


"""




from flax import linen as nn
import functools

import jax
import jax.numpy as jnp
import jax.tree_util
from jaxtyping import Array

import logging

import math
import matplotlib.pyplot as plt

from numpy.typing import NDArray

import os
import re

from tqdm import tqdm


from src.config_classes.ddpm_config import DDPMConfig
from src.config_classes.optimizer_config import OptimizerConfig
from src.config_classes.training_config import TrainingConfig

from src.evaluation import display

from src.file_handling import naming, save_load_hdf5

from src.model_handling import ModelHandler

from src.training import multi_gpu_util



class Sampler(ModelHandler):
    
    def __init__(self,
                 train_config: TrainingConfig, 
                 model_config: DDPMConfig,
                 optimizer_config: OptimizerConfig):
        
        super().__init__(train_config, model_config, optimizer_config)
        
    
        
    def draw_samples(self, 
                     num_samples: int, 
                     seed: int, 
                     save_to_folder: str,
                     date_str: str,
                     step: int,
                     reverse_noise_from_test_samples: bool,
                     num_classes: int):
        sample_dir = os.path.join(save_to_folder, 'samples')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        
        model_name = naming.get_model_name(self.model_config, self.train_config, self.optimizer_config)
        prefix = naming.get_file_prefix_from_name(model_name, date_str)
        self.train_state = self.load_checkpoint(save_to_folder, prefix, step)
        
        grid_size = int(jnp.floor(jnp.sqrt(num_samples)))
        
        all_samples, all_names = self.get_samples_as_matrix(
            num_samples, seed, date_str, step, 
            reverse_noise_from_test_samples, 
            num_classes)        
        
        for i in range(len(all_samples)): 
            if all_names[i] == 'x':
                current_samples = all_samples[i].astype(int)
            else:
                current_samples = all_samples[i]
        
            display.make_image_grid(
                current_samples[:grid_size**2], grid_size, grid_size, model_name)
                
            if save_to_folder == '':
                plt.show()
            else:
                steps = self.train_state.step
                sample_file_name = f'{prefix}sample_{steps}_{all_names[i]}.png'
                sample_path = os.path.join(sample_dir, sample_file_name)
                plt.savefig(sample_path, dpi = 300)
                plt.clf()
        
        return all_samples
    
    
    def save_samples_as_h5(self,
                            num_samples: int, 
                            seed: int,
                            save_to_folder: str,
                            date_str: str,
                            step: int,
                            use_T: int = 0,
                            use_dynamic_thresholding: bool = False,
                            dt_percentile: float = 0.0,
                            percentile_scale: float = 0.0,
                            init_var_bound: float = 0.0) -> None:
        sample_dir = os.path.join(save_to_folder, 'samples')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)        
        
        h5_dataset_name = naming.get_samples_h5_dataset_name()
        
        model_name = naming.get_model_name(self.model_config, self.train_config, self.optimizer_config)
        prefix = naming.get_file_prefix_from_name(model_name, date_str)
        
        self.train_state = self.load_checkpoint(save_to_folder, prefix, step)
        
        if num_samples < 100:
            batch_size = num_samples
        else:
            batch_size = 100
        batches = math.ceil(num_samples/batch_size)
        
        sample_file_prefix = naming.get_sample_file_prefix(prefix, step, batch_size)
        sample_path_prefix = os.path.join(sample_dir, sample_file_prefix)
        
        if use_T > 0:
            T_suffix = f'_T{use_T}' 
        else: 
            T_suffix = ''
        
        if use_dynamic_thresholding:
            dt_suffix = f'_dt{str(dt_percentile).replace(".", "_")}' 
        else: 
            dt_suffix = ''
        
        if percentile_scale > 0:
            psc_suffix = f'_psc{str(percentile_scale).replace(".", "_")}' 
        else: 
            psc_suffix = ''
        
        if init_var_bound > 0:
            init_suffix = f'_initvar{str(init_var_bound).replace(".", "_")}' 
        else: 
            init_suffix = ''
        
        sample_files = [file 
                        for file in os.listdir(sample_dir)
                        if re.match(f'{sample_file_prefix}{T_suffix}{dt_suffix}{psc_suffix}{init_suffix}_....h5', file)]
        if sample_files:
            batch_numbers = [int(file.split('_')[-1].split('.')[0])
                             for file in sample_files]
        else:
            batch_numbers = [-1]
                
        if max(batch_numbers) < batches - 1:
            for i in range(batches):
                if i in batch_numbers:
                    continue 
                
                all_samples, _ = self.get_samples_as_matrix(
                    batch_size, seed + i, date_str, step, 
                    False, 
                    10,
                    use_T,
                    use_dynamic_thresholding,
                    dt_percentile,
                    percentile_scale,
                    init_var_bound)
                                
                save_load_hdf5.save_to_hdf5(
                    all_samples, 
                    h5_file_path = f'{sample_path_prefix}{T_suffix}{dt_suffix}{psc_suffix}{init_suffix}_{i:03d}.h5', 
                    h5_dataset_name = h5_dataset_name)
        
    
    def get_samples_as_matrix(self,
                              num_samples: int, 
                              seed: int,
                              date_str: str,
                              step: int,
                              reverse_noise_from_test_samples: bool,
                              num_classes: int,
                              use_T: int = 0,
                              use_dynamic_thresholding: bool = False,
                              dt_percentile: float = 0.0,
                              percentile_scale: float = 0.0,
                              init_var_bound: float = 0.0):
        
        model_name = naming.get_model_name(self.model_config, self.train_config, self.optimizer_config)
                  
        sample_rng = jax.random.PRNGKey(seed)
        
        if '_conditional_' in model_name:
            classes = jnp.arange(0, 10)
            repeats = math.ceil(num_samples / len(classes))
            conditions = jnp.tile(classes, repeats)[:num_samples]
        else:
            conditions = jnp.array([])
            
        if use_T > 0:
            T = use_T
        else:        
            timesteps = self.model_config.n_timesteps
            if timesteps > 0:
                T = timesteps
            else:
                T = 1000
        
        if reverse_noise_from_test_samples:
            examples = self.get_test_examples(num_samples, num_classes)
            z, x_sample = self.reverse_noise_from_examples(
                sample_rng,
                examples,
                T_sample = T,
                conditions = conditions)
        
        elif 'SVB_' in model_name:
            z, x_sample = self.bootstrap_sample_from_model(
                sample_rng, num_samples, T, conditions,
                use_dynamic_thresholding, dt_percentile,
                percentile_scale)
        else:        
            z, x_sample = self.sample_from_model(
                sample_rng, num_samples, T, conditions,
                use_dynamic_thresholding, dt_percentile,
                percentile_scale, init_var_bound)
        
        
        intermediate_sample_indexes = jnp.arange(0, T + 1, 100)
        intm_samples = jnp.array(z)[intermediate_sample_indexes, :, :, :]
        intm_sample_names = [f'z_{i}' for i in intermediate_sample_indexes]
               
        samples_mins = jnp.min(intm_samples, axis = [1,2,3,4])
        move_to_zero_diff = jnp.minimum(samples_mins, 0).reshape(*samples_mins.shape, 1, 1, 1, 1)

        moved_samples = intm_samples + jnp.abs(move_to_zero_diff)

        moved_max = jnp.max(moved_samples, axis = [1,2,3,4]).reshape(*samples_mins.shape, 1, 1, 1, 1)
        intm_samples = moved_samples/moved_max
       
        all_samples = jnp.vstack((intm_samples, x_sample.reshape(1, *x_sample.shape)))
        all_names = intm_sample_names + ['x']
        
        return all_samples, all_names
    
    
    def sample_from_model(self, 
                          rng, 
                          N_sample: int, 
                          T_sample: int,
                          conditions: NDArray[int],
                          use_dynamic_thresholding: bool = False,
                          dt_percentile: float = 0.0,
                          percentile_scale: float = 0.0,
                          init_var_bound: float = 0.0):
                
        model = self.model 
        state = self.train_state
        if len(conditions) > 0 and len(conditions) != N_sample:
            raise ValueError('Number of conditions do not match samples')
        if len(conditions) == 0:
            conditions = jnp.zeros((N_sample,))
        
        
        # sample z_1 for the diffusion model
        rng, rng1 = jax.random.split(rng)
                
        height = self.input_shape[0]
        width = self.input_shape[1]
        channels = self.input_shape[2]
        z = [jax.random.normal(rng1, 
                               (N_sample, height, width, channels))]
        
        if init_var_bound > 0:
            z_var = jnp.var(z[0])
            mult = jnp.sqrt(init_var_bound / jnp.maximum(z_var, init_var_bound)) 
            
            z[0] = z[0] * mult
                        
        
        for i in tqdm(range(T_sample)):
            rng, rng1 = jax.random.split(rng)
            _z = state.apply_fn(
                variables = {'params': state.ema_params},
                i = i, 
                T = T_sample, 
                z_t = z[-1], 
                conditioning = conditions, 
                rng = rng1,
                dynamic_thresholding = use_dynamic_thresholding,
                percentile = dt_percentile,
                percentile_scale = percentile_scale, 
                method = model.sample)
            z.append(_z)
        
        x_sample = state.apply_fn(
            {'params': state.ema_params}, z[-1], method=model.generate_x)
        
        return z, x_sample
    
    
    def save_samples_as_h5_multi(self,
                            num_samples: int, 
                            seed: int,
                            save_to_folder: str,
                            date_str: str,
                            step: int,
                            idx_to_keep: Array,
                            batch_size: int,
                            use_T: int = 0,
                            use_dynamic_thresholding: bool = False,
                            dt_percentile: float = 0.0,
                            percentile_scale: float = 0.0) -> None:
        sample_dir = os.path.join(save_to_folder, 'samples')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)        
        
        h5_dataset_name = naming.get_samples_h5_dataset_name()
        
        model_name = naming.get_model_name(self.model_config, self.train_config, self.optimizer_config)
        prefix = naming.get_file_prefix_from_name(model_name, date_str)
        
        if 'SVB_' in model_name:
            raise NotImplementedError('SVB model sampling not implemented for multi gpu')
        
        self.train_state = self.load_checkpoint(save_to_folder, prefix, step)
        
        if batch_size == 0:
            batch_size = self.train_config.batch_size
        batches = math.ceil(num_samples/batch_size)
        
        device_count = jax.local_device_count()
        if batch_size % device_count != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by the number of devices ({device_count}).")
        
        logging.info(f'Devices: {device_count}')
        self.device_count = device_count
        
        sample_file_prefix = naming.get_sample_file_prefix(prefix, step, batch_size)
        sample_path_prefix = os.path.join(sample_dir, sample_file_prefix)
        
        if use_T > 0:
            T_suffix = f'_T{use_T}' 
        else: 
            T_suffix = ''
        
        if use_dynamic_thresholding:
            dt_suffix = f'_dt{str(dt_percentile).replace(".", "_")}' 
        else: 
            dt_suffix = ''
        
        if percentile_scale > 0:
            psc_suffix = f'_psc{str(percentile_scale).replace(".", "_")}' 
        else: 
            psc_suffix = ''
                
        
        sample_files = [file 
                        for file in os.listdir(sample_dir)
                        if re.match(f'{sample_file_prefix}{T_suffix}{dt_suffix}{psc_suffix}_....h5', file)]
        if sample_files:
            batch_numbers = [int(file.split('_')[-1].split('.')[0])
                             for file in sample_files]
            logging.info(f'batch_numbers: {batch_numbers}')
        else:
            batch_numbers = [-1]
                
        if max(batch_numbers) < batches - 1:
            
            if use_T > 0:
                T = use_T
            else:        
                timesteps = self.model_config.n_timesteps
                if timesteps > 0:
                    T = timesteps
                else:
                    T = 1000
            
            if len(idx_to_keep) == 0:
                idx_to_keep = jnp.arange(0, T + 1, 100)
                idx_to_keep = idx_to_keep.at[-1].set(idx_to_keep[-1] - 1)
            
            height = self.input_shape[0]
            width = self.input_shape[1]
            channels = self.input_shape[2]
            dummy_inputs = jnp.zeros(
                (self.device_count, int(batch_size/self.device_count), 
                 len(idx_to_keep) + 1, 
                 height, width, channels))
            
                
            if '_conditional_' in model_name:
                raise NotImplementedError('Conditional model not implemented')
                classes = jnp.arange(0, 10)
                repeats = math.ceil(num_samples / len(classes))
                conditions = jnp.tile(classes, repeats)[:num_samples]
            else:
                conditions = jnp.array([])
            
            if len(conditions) > 0 and len(conditions) != num_samples:
                raise ValueError('Number of conditions do not match samples')
            if len(conditions) == 0:
                conditions = jnp.zeros(
                    (self.device_count, int(batch_size/self.device_count)))
            
            for i in tqdm(range(batches)):
                if i in batch_numbers:
                    continue 
                
                sample_rng = jax.random.PRNGKey(seed + i)
                
                self.p_sample = functools.partial(
                    self.sample_from_model_multi,
                    rng = sample_rng,                    
                    T_sample = T,                    
                    idx_to_keep = idx_to_keep,
                    use_dynamic_thresholding = use_dynamic_thresholding,
                    dt_percentile = dt_percentile,
                    percentile_scale = percentile_scale)
                
                self.p_sample = multi_gpu_util.dist(
                    self.p_sample, accumulate='concat', axis_name='devices')
                
                
                samples = self.get_samples_as_matrix_multi(dummy_inputs, conditions)
                                
                save_load_hdf5.save_to_hdf5(
                    samples, 
                    h5_file_path = f'{sample_path_prefix}{T_suffix}{dt_suffix}{psc_suffix}_{i:03d}.h5', 
                    h5_dataset_name = h5_dataset_name)
    
    
    def get_samples_as_matrix_multi(self, dummy_inputs: Array, conditions: Array):
        samples = self.p_sample(dummy_inputs = dummy_inputs, conditions = conditions)
        sample_types = samples.shape[1]
        reshaped_samples = jnp.zeros((sample_types, samples.shape[0], *samples.shape[2:]))
        
        
        for i in range(sample_types):
            reshaped_samples = reshaped_samples.at[i].set(samples[:, i])
        samples = reshaped_samples
        
        
        intm_samples = samples[:-1]
        samples_mins = jnp.min(intm_samples, axis = [1,2,3,4])
        samples_mins = jnp.minimum(samples_mins, 0).reshape(*samples_mins.shape, 1, 1, 1, 1)
        
        intm_samples = intm_samples + jnp.abs(samples_mins)
        moved_max = jnp.max(intm_samples, axis = [1,2,3,4]).reshape(*samples_mins.shape)
        intm_samples = intm_samples/moved_max
        
        samples = samples.at[:-1].set(intm_samples)
        
        return samples
    
    
    # See https://github.com/google-research/vdm/blob/main/experiment_vdm.py
    def sample_from_model_multi(self, 
                          rng, 
                          dummy_inputs: Array, 
                          T_sample: int,
                          conditions: NDArray[int],
                          idx_to_keep: Array,
                          use_dynamic_thresholding: bool = False,
                          dt_percentile: float = 0.0,
                          percentile_scale: float = 0.0):
        rng = jax.random.fold_in(rng, jax.lax.axis_index('devices'))
        
        model = self.model
        state = self.train_state
        
        # sample z_1 for the diffusion model
        rng, rng1 = jax.random.split(rng)
        
        z_1 = jax.random.normal(rng1, dummy_inputs[:, 0].shape)
        samples = jnp.zeros(dummy_inputs.shape)
        
        samples = samples.at[:, -1].set(z_1)
        
        def inner_fn(i, samples):
            z_i = state.apply_fn(
                variables = {'params': state.ema_params},
                i = i, 
                T = T_sample, 
                z_t = samples[:, -1], 
                conditioning = conditions, 
                rng = rng,
                dynamic_thresholding = use_dynamic_thresholding,
                percentile = dt_percentile,
                percentile_scale = percentile_scale, 
                method = model.sample)
            
            # If i is not one of the indexes to keep, idx is set to -1
            # So in this case, samples.at[:, -1] is set twice
            idx = jnp.nonzero(idx_to_keep == i, size = 1, fill_value = -1)[0][0]
            samples = samples.at[:, idx].set(z_i)
            
            samples = samples.at[:, -1].set(z_i)
            return samples
        
        samples = jax.lax.fori_loop(
            lower=0, upper=T_sample, body_fun=inner_fn, init_val=samples)
        
        x_sample = state.apply_fn(
            {'params': state.ema_params}, samples[:, -1], method=model.generate_x)
        
        samples = samples.at[:, -1].set(x_sample)
                
        return samples
    
    
    def bootstrap_sample_from_model(self, 
                          rng, 
                          N_sample: int, 
                          T_sample: int,
                          conditions: NDArray[int],
                          use_dynamic_thresholding: bool = False,
                          dt_percentile: float = 0.0,
                          percentile_scale: float = 0.0):
        model = self.model 
        state = self.train_state
        if len(conditions) > 0 and len(conditions) != N_sample:
            raise ValueError('Number of conditions do not match samples')
        if len(conditions) == 0:
            conditions = jnp.zeros((N_sample,))
        
        
        # sample z_1 from the diffusion model
        rng, rng1 = jax.random.split(rng)
                
        height = self.input_shape[0]
        width = self.input_shape[1]
        channels = self.input_shape[2]
        z = [jax.random.normal(rng1, 
                               (N_sample, height, width, channels))]
        z.append(jax.random.normal(rng1, 
                               (N_sample, height, width, channels)))
                                
        
        for i in tqdm(range(T_sample)):
            rng, rng1 = jax.random.split(rng)
            _z = state.apply_fn(
                variables= state.params,
                i = i, 
                T = T_sample, 
                z_t = z[-1],
                z_u = z[-2],
                conditioning = conditions, 
                rng = rng1,
                dynamic_thresholding = use_dynamic_thresholding,
                percentile = dt_percentile,
                percentile_scale = percentile_scale, 
                method = model.sample)
            z.append(_z)
        
        x_sample = state.apply_fn(state.params, z[-1], method=model.generate_x)
        
        return z, x_sample
    
    
    def get_test_examples(self, N_sample: int, num_classes: int):
        batch = next(iter(self.eval_ds))
    
        images, conditioning = batch
        
        ex_per_class = math.floor(N_sample / num_classes)
        real_num_examples = ex_per_class * num_classes
        
        examples = jnp.zeros((real_num_examples, images.shape[1], images.shape[2], images.shape[3]))
        
        classes = jnp.arange(0, num_classes)
        for i in range(num_classes):
            filt = conditioning == classes[i]
            examples[i*ex_per_class: (i+1)*ex_per_class] = images[filt][0:ex_per_class]
        
        return examples        
    
    
    def reverse_noise_from_examples(self, 
                          rng, 
                          samples: Array, 
                          T_sample: int,
                          conditions: Array):
        model = self.model 
        state = self.train_state        
        height = self.input_shape[0]
        width = self.input_shape[1]
        channels = self.input_shape[2]
        N_sample = samples.shape[0]
        
        if len(conditions) > 0 and len(conditions) != N_sample:
            raise ValueError('Number of conditions do not match samples')
        if len(conditions) == 0:
            conditions = jnp.zeros((N_sample,))
        
        t_1 = jnp.ones((N_sample,))
        
        x_1 = state.apply_fn(
            variables= state.params,
            x = samples, 
            t = t_1,
            conditioning = conditions,
            deterministic = True, 
            method = model.encode)
        
        # z_1 = alpha_1*x_1 + sigma_1*eps
        rng, rng1 = jax.random.split(rng)
        eps = jax.random.normal(rng1, 
                               (N_sample, height, width, channels))
        g_1 = state.apply_fn(
            variables= state.params,
            t = t_1,            
            method = model.get_gamma)
        g_1 = g_1.reshape(-1, 1, 1, 1)
        var_1 = nn.sigmoid(g_1)
        alpha_1 = jnp.sqrt(1-var_1)
        sigma_1 = jnp.sqrt(var_1)
        z_1 = alpha_1 * x_1 + sigma_1 * eps 
        
        z = [z_1]
        
        for i in tqdm(range(T_sample)):
            rng, rng1 = jax.random.split(rng)
            _z = state.apply_fn(
                variables= state.params,
                i = i, 
                T = T_sample, 
                z_t = z[-1], 
                conditioning = conditions, 
                rng = rng1, 
                method = model.sample)
            z.append(_z)
        
        x_sample = state.apply_fn(state.params, z[-1], method=model.generate_x)
        
        return z, x_sample
    
    
    def for_test_z_t_vis(self, intm_samples: Array):
        model = self.model 
        state = self.train_state 
        for i in range(len(intm_samples)):
            intm_samples = intm_samples.at[i].set(
                state.apply_fn(
                variables= state.params,
                z_t = intm_samples[i],
                method = model.z_t_for_visualisation))
        
        return intm_samples
    
    
    def load_model_for_sampler(self, 
                               save_to_folder: str, 
                               date_str: str,
                               step: int,):
        prefix = naming.get_file_prefix(
            self.model_config, self.train_config, self.optimizer_config, date_str)
                
        self.train_state = self.load_checkpoint(save_to_folder, prefix, step)
    
           
        
    
    
    
    
    
    
    
    
    



