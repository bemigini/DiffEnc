#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:37:05 2022

@author: bemi


Training pipeline


Much of this code is based on 
https://github.com/google-research/vdm/blob/main/experiment.py
and
https://github.com/google-research/vdm/blob/main/experiment_vdm.py


"""



from datetime import date

from flax.training import checkpoints
from flax.training.common_utils import stack_forest

import flax.jax_utils as flax_utils
#from flax.training.train_state import TrainState

import jax
import jax.numpy as jnp
from jax._src.random import PRNGKey
import jax.tree_util 
from jaxtyping import Array

import logging
import math

import os

import shutil

from tqdm import tqdm


from src.config_classes.ddpm_config import DDPMConfig
from src.config_classes.optimizer_config import OptimizerConfig
from src.config_classes.training_config import TrainingConfig

from src.data import load_data 

from src.file_handling import naming, save_load_json
from src.model_handling import ModelHandler

from src.training.train_state import EMATrainState


class TrainingPipeline(ModelHandler):
        
    def __init__(self, 
                 train_config: TrainingConfig, 
                 model_config: DDPMConfig,
                 optimizer_config: OptimizerConfig,
                 output_folder: str,
                 date_str: str = '',
                 imagenet_folder: str = ''):
        logging.warning('=== Initializing training pipeline ===')
        
        super().__init__(train_config, model_config, optimizer_config)
        
        self.output_folder = output_folder
        checkpoint_dir = os.path.join(output_folder, 'checkpoints')
        metrics_dir = os.path.join(output_folder, 'metrics')
        keep_dir = os.path.join(output_folder, 'keep')
        if not os.path.exists(checkpoint_dir):
           os.mkdir(checkpoint_dir) 
        if not os.path.exists(metrics_dir):
           os.mkdir(metrics_dir) 
        if not os.path.exists(keep_dir):
           os.mkdir(keep_dir) 
        
        # Initialize train and eval rng
        self.rng, train_rng = jax.random.split(self.rng)
        self.train_rng = train_rng
        self.rng, eval_rng = jax.random.split(self.rng)
        self.eval_rng = eval_rng
                      
        batch_size = self.train_config.batch_size
        device_count = jax.local_device_count()
        if batch_size % device_count != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by the number of devices ({device_count}).")
        
        logging.info(f'Devices: {device_count}')
        self.device_count = device_count
        
        self.train_ds, self.eval_ds, self.condition_classes = load_data.load_dataset(
            self.train_config, imagenet_folder)
        
        self.eval_steps = load_data.get_eval_steps(
            self.train_config.dataset_name, self.train_config.batch_size)
        
        self.train_metrics = []
        
                
        if date_str != '':
            restored_state, prefix = self.load_checkpoint_date(output_folder, date_str)
            self.train_state = restored_state
                        
            restored_metrics = self.load_metrics(output_folder, prefix)
            self.train_metrics = restored_metrics            
            
        else:
            today = date.today()
            date_str = str(today)
            prefix = naming.get_file_prefix(self.model_config, self.train_config, self.optimizer_config, date_str)
                    
        self.prefix = prefix
        
        # If we want to use sub-train steps
        # self.p_train_step = functools.partial(jax.lax.scan, self.train_step_multi_gpu)
        self.p_train_step = jax.pmap(self.train_step_multi_gpu, 'devices')
        self.p_eval_step = jax.pmap(self.eval_step_multi_gpu, 'devices')
                    
        
        logging.warning('=== Done initializing training pipeline ===')
    
    
    def train_step_multi_gpu(
            self,
            state: EMATrainState, 
            im_batch: Array,
            cond_batch: Array):
                    
        rng1 = jax.random.fold_in(self.train_rng, jax.lax.axis_index('devices'))
        rng1 = jax.random.fold_in(rng1, state.step)
                
        grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
        (loss, metrics), grads = grad_fn(
            state.params, im_batch, cond_batch, state, rng1, is_train = True)
        
        grads = jax.lax.pmean(grads, axis_name = 'devices')
                
        learning_rate = self.lr_schedule(state.step)
        new_state = state.apply_gradients(
            grads = grads, 
            lr = learning_rate, 
            ema_rate = self.optimizer_config.ema_rate)
            
        metrics['scalars'] = jax.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='devices'), metrics['scalars'])
                        
        return new_state, metrics
    
        
    def eval_step_multi_gpu(self, 
                            state: EMATrainState, 
                            im_batch: Array,
                            cond_batch: Array,
                            eval_step=0):
        rng = jax.random.fold_in(self.eval_rng, jax.lax.axis_index('devices'))
        rng = jax.random.fold_in(rng, eval_step)
                
        _, metrics = loss_fn(state.ema_params, 
                             im_batch, 
                             cond_batch, 
                             state, rng=rng, is_train=False)
        
        metrics['scalars'] = jax.tree_map(
            lambda x: jax.lax.pmean(x, axis_name='devices'), metrics['scalars'])
        
        return metrics
    
    
    # Multi-gpu distribution following: 
    # https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html
    # and
    # https://github.com/google-research/vdm/blob/main/experiment.py
    def train_multi_gpu(self): 
        train_config = self.train_config
        output_folder = self.output_folder
        prefix = self.prefix
        dataset_name = train_config.dataset_name
        
        logging.info(f'Saving in folder: {output_folder}')
        
        metrics_dir = os.path.join(output_folder, 'metrics')           
        
        logging.info('num_steps_train=%d', train_config.num_steps_train)
        saves_str = [str(s) for s in train_config.step_saves_to_keep]
        logging.info(f'Steps to keep: {",".join(saves_str)}')
        
        state = self.train_state
        state = flax_utils.replicate(state)
        
        checkpoint_dir = os.path.join(output_folder, 'checkpoints')
        keep_dir = os.path.join(output_folder, 'keep')
        
        train_steps = train_config.num_steps_train
        nan_found = False
        
        if 'imagenet' not in dataset_name:            
            for i in tqdm(range(1, train_steps + 1)):
                
                state, cont_train = self.training_and_logging_iteration(
                    i, self.train_ds, dataset_name, state, 
                    train_config, train_steps, nan_found,
                    checkpoint_dir, prefix, keep_dir,
                    metrics_dir)                 
                
                if not cont_train:
                    break
        
        else:
            steps_per_file = math.floor(128116/train_config.batch_size)
            for j in tqdm(range(math.ceil(train_steps/steps_per_file))):
                train_ds_file_it = iter(next(self.train_ds))
                
                for s in range(steps_per_file):
                    i = (j*steps_per_file + s + 1)
                    
                    state, cont_train = self.training_and_logging_iteration(
                        i, train_ds_file_it, dataset_name, state, 
                        train_config, train_steps, nan_found,
                        checkpoint_dir, prefix, keep_dir,
                        metrics_dir)
                    
                    if not cont_train:
                        break
                
                if not cont_train:
                    break
        
        self.train_state = flax_utils.unreplicate(state)
        
        

    def training_and_logging_iteration(self, i, train_ds, dataset_name, state, 
                                       train_config, train_steps, nan_found,
                                       checkpoint_dir, prefix, keep_dir,
                                       metrics_dir):
        jnp_batch = jax.tree_util.tree_map(jnp.asarray, next(train_ds))
        get_device_batch = lambda x: x.reshape(self.device_count, -1, *x.shape[1:])
        jnp_batch = jax.tree_util.tree_map(get_device_batch, jnp_batch)
        im_batch, cond_batch = jnp_batch
        if self.condition_classes == 0:
            cond_batch = jnp.zeros(cond_batch.shape)
        
        
        if len(self.train_metrics) == 0:
            init_metrics = self.p_eval_step(state, im_batch, cond_batch)
            
            init_metrics = {'train': flax_utils.unreplicate(init_metrics['scalars'])}
            self.train_metrics.append({0: init_metrics})
        
        state, train_metrics = self.p_train_step(state, im_batch, cond_batch)
        
        new_step = int(state.step[0])
        step = new_step
    
        if i == 1 or i % train_config.steps_per_logging == 0 or i == train_steps:
            metrics = flax_utils.unreplicate(train_metrics['scalars'])
            t_metrics = {'train': metrics}
            nan_found = nan_found or any([jnp.isnan(jnp.array(val)).any()
                                          for val 
                                          in metrics.values()])                
            step_metrics = {step: t_metrics}
            logging.info(step_metrics)
            self.train_metrics.append(step_metrics)
            
        if i % train_config.steps_per_save == 0 or i == train_steps:
            unrep_state = flax_utils.unreplicate(state)
            checkpoints.save_checkpoint(
                ckpt_dir = checkpoint_dir, 
                target = unrep_state,
                step = step,
                prefix = prefix,
                keep = 3,
                overwrite = True)
            
            if step in train_config.step_saves_to_keep:
                src_path = os.path.join(checkpoint_dir, f'{prefix}{step}')
                shutil.copy2(src_path, keep_dir)
            
        
        if i % train_config.steps_per_eval == 0 or i == train_steps:
            iter_eval = iter(self.eval_ds)
            
            if 'imagenet' in dataset_name:
                iter_eval = iter(next(iter_eval))
            
            eval_metrics = []
            eval_it = 0
            for e_batch in iter_eval:
                eval_batch = jax.tree_util.tree_map(jnp.asarray, e_batch)
                eval_batch = jax.tree_util.tree_map(get_device_batch, eval_batch)
                im_batch, cond_batch = eval_batch
                if self.condition_classes == 0:
                    cond_batch = jnp.zeros(cond_batch.shape)
                    
                metrics = self.p_eval_step(state, im_batch, cond_batch, flax_utils.replicate(eval_it))
                eval_it += 1
                metrics = flax_utils.unreplicate(metrics['scalars'])
                eval_metrics.append(metrics)
                
            mean_eval_metrics = stack_forest(eval_metrics)
            mean_eval_metrics = jax.tree_map(jnp.mean, mean_eval_metrics)            
            if step in self.train_metrics[-1].keys():
                self.train_metrics[-1][step]['eval'] = mean_eval_metrics
            else:
                e_metrics = {'eval': mean_eval_metrics}
                step_metrics = {step: e_metrics}
                self.train_metrics.append(step_metrics)
        
        if i % train_config.steps_per_eval == 0 or i % train_config.steps_per_logging == 0 or i == train_steps:
            metrics_path = os.path.join(metrics_dir, f'{prefix}train_metrics.json')
            save_load_json.save_as_json(self.train_metrics, metrics_path)
            # nan check                
            if nan_found:
                logging.warning('Nan found. Stopping training')
                return state, False
        
        return state, True


def loss_fn(params, images: Array, conditioning: Array, 
            state: EMATrainState, rng: PRNGKey, is_train: bool,
            ts: Array = jnp.array([])):
    rng, sample_rng = jax.random.split(rng)
    rng, encode_rng = jax.random.split(rng)
    rngs = {'sample': sample_rng, 'encode': encode_rng}
    if is_train:
        rng, dropout_rng = jax.random.split(rng)
        rngs['dropout'] = dropout_rng
        
    # sample time steps 
    outputs = state.apply_fn(
        variables = {'params': params},
        images = images,
        conditioning = conditioning,
        rngs = rngs,
        deterministic = not is_train,
        ts = ts)
    
    rescale_to_bpd = 1./(jnp.prod(jnp.array(images.shape[1:])) * jnp.log(2.))
    bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd
    bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd
    bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd
    loss = bpd_recon + bpd_latent + bpd_diff
            
    g_mmm = outputs.gamma_grad_min_mean_max
    x_t_mmm = outputs.x_t_grad_min_mean_max
    inv_loss = jnp.mean(outputs.inv_loss) * rescale_to_bpd
    
    t_diff_loss = [jnp.mean(t_o) * rescale_to_bpd for t_o in outputs.t_diff_loss]
        
    scalar_dict = {
        "bpd": loss,
        "bpd_latent": bpd_latent,
        "bpd_recon": bpd_recon,
        "bpd_diff": bpd_diff,
        "var0": outputs.var_0,
        "var": outputs.var_1,
        "gamma_grad_min_mean_max": g_mmm,
        "SNR_t_min_mean_max": outputs.SNR_t_min_mean_max,
        "SNR_t_times_x_t_grad_norm_min_mean_max": outputs.SNR_t_times_x_t_grad_norm_min_mean_max,
        "eps_m_eps_hat_norm_min_mean_max": outputs.eps_m_eps_hat_norm_min_mean_max,
        "t_diff_loss": t_diff_loss,
        "x_t_grad_min_mean_max": x_t_mmm,
        "non_zero_x_t_grad": outputs.non_zero_x_t_grad,
        "inv_loss": inv_loss
    }
    # no longer used: img_dict = {"inputs": images}
    # no longer used:metrics = {"scalars": scalar_dict, "images": img_dict}
    metrics = {"scalars": scalar_dict}
    
    return loss, metrics







