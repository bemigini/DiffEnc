#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:41:16 2023

@author: bemi


Base class for classes which need to handle models


Much of this code is based on 
https://github.com/google-research/vdm/blob/main/experiment.py
and
https://github.com/google-research/vdm/blob/main/experiment_vdm.py


"""



import flax
from flax.core.frozen_dict import unfreeze, FrozenDict
from flax.training import checkpoints
from flax.training.train_state import TrainState


import jax
import jax.numpy as jnp
from jax._src.random import PRNGKey
import jax.tree_util 

import optax
from optax._src import base
import os

from tqdm import tqdm

from typing import Dict, List, Tuple

from src.basic_ddpm.vdm import VDM2
from src.basic_ddpm.vdm_v import VDM2V
from src.config_classes.ddpm_config import DDPMConfig
from src.config_classes.optimizer_config import OptimizerConfig
from src.config_classes.training_config import TrainingConfig

from src.data import load_data 

from src.file_handling import naming, save_load_json

from src.training import train_state


from src.learned_encoder.sigma3_v_approx_model import Sigma3VA
from src.learned_encoder.sigma4_v_nontrainable_enc_model import Sigma4VNT


class ModelHandler():
    
    def __init__(self, 
                 train_config: TrainingConfig, 
                 model_config: DDPMConfig,
                 optimizer_config: OptimizerConfig):
        
        self.train_config = train_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        
        seed = self.train_config.seed 
        self.rng = jax.random.PRNGKey(seed)
        
        self.input_shape = load_data.get_input_shape(self.train_config.dataset_name)
                
        # Initialize model
        self.rng, model_rng = jax.random.split(self.rng)
        self.model_rng = model_rng
        self.model, params = get_model_and_params(
            self.model_config, model_rng, self.input_shape)
        
        # Create train state
        self.train_state = train_state.EMATrainState.create(
            apply_fn = self.model.apply,
            variables = params,
            optax_optimizer = self.get_optimizer)
        self.lr_schedule = self.get_lr_schedule()
    
    
    def get_optimizer(self, lr: float) -> base.GradientTransformation:
        config = self.optimizer_config
        
        def decay_mask_fn(params):
          flat_params = flax.traverse_util.flatten_dict(unfreeze(params))
          flat_mask = {
              path: (path[-1] != "bias" and path[-2:]
                     not in [("layer_norm", "scale"), ("final_layer_norm", "scale")])
              for path in flat_params
          }
          return FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))

        if config.name == "adamw":
          optimizer = optax.adamw(
              learning_rate = lr,
              mask = decay_mask_fn,
              b1 = config.args.b1,
              b2 = config.args.b2,
              eps = config.args.eps,
              weight_decay = config.args.weight_decay
          )
          if config.use_gradient_clipping:
            clip = optax.clip_by_global_norm(config.gradient_clip_norm)
            optimizer = optax.chain(clip, optimizer)
            print(f'using gradient clipping: {config.gradient_clip_norm}')
        else:
          raise Exception('Unknow optimizer.')

        return optimizer
    
    
    def get_lr_schedule(self):
        learning_rate = self.optimizer_config.learning_rate
        config_train = self.train_config
        # Create learning rate schedule
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=config_train.num_steps_lr_warmup
        )
    
        if self.optimizer_config.lr_decay:
          decay_fn = optax.linear_schedule(
              init_value=learning_rate,
              end_value=0,
              transition_steps=config_train.num_steps_train - config_train.num_steps_lr_warmup,
          )
          schedule_fn = optax.join_schedules(
              schedules=[warmup_fn, decay_fn], boundaries=[
                  config_train.num_steps_lr_warmup]
          )
        else:
          schedule_fn = warmup_fn
    
        return schedule_fn
        
    
    # define sampling function
    def sample_from_model(self, rng, N_sample: int, T_sample: int):
        model = self.model 
        state = self.train_state
        # sample z_0 from the diffusion model
        rng, rng1 = jax.random.split(rng)
                
        height = self.input_shape[0]
        width = self.input_shape[1]
        channels = self.input_shape[2]
        z = [jax.random.normal(rng1, 
                               (N_sample, height, width, channels))]
        
        for i in tqdm(range(T_sample)):
            rng, rng1 = jax.random.split(rng)
            _z = state.apply_fn(
                variables= state.ema_params,
                i = i, 
                T = T_sample, 
                z_t = z[-1], 
                conditioning = jnp.zeros((z[-1].shape[0],)), 
                rng = rng1, 
                method = model.sample)
            z.append(_z)
        
        x_sample = state.apply_fn(state.ema_params, z[-1], method=model.generate_x)
        
        return z, x_sample
    
    
    def load_checkpoint_date(self, 
                             output_folder: str, 
                             date_str: str) -> Tuple[TrainState, str]:
        prefix = naming.get_file_prefix(self.model_config, self.train_config, self.optimizer_config, date_str)
                
        return self.load_checkpoint(output_folder, prefix), prefix
    
    
    def load_checkpoint(self, output_folder: str, prefix: str, step: int = None) -> TrainState:
        checkpoint_dir = os.path.join(output_folder, 'checkpoints')
                
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir = checkpoint_dir, 
            target = self.train_state, 
            step = step,
            prefix = prefix)
        
        if restored_state.step == 0:
            if step is not None: 
                step_suffix = ' and step: {step}'
            else: 
                step_suffix = ''
                
            raise ValueError(f'No trained checkpoint found with prefix: {prefix}{step_suffix}')
                
        return restored_state
    
    
    def load_metrics_date(self, 
                          output_folder: str, 
                          date_str: str) -> Tuple[List[Dict], str]:
        prefix = naming.get_file_prefix(self.model_config, self.train_config, self.optimizer_config, date_str)
        
        return self.load_metrics(output_folder, prefix), prefix
    
    
    def load_metrics(self, output_folder: str, prefix: str) -> List[Dict]:
        metrics_dir = os.path.join(output_folder, 'metrics')    
        metrics_path = os.path.join(metrics_dir, f'{prefix}train_metrics.json')
        restored_metrics = save_load_json.load_json(metrics_path)
        
        return restored_metrics


def get_model_and_params(
        config: DDPMConfig, 
        rng: PRNGKey,
        input_shape: Tuple[int, int, int]):
    if config.ddpm_type == 'VDM2':
        model = VDM2(config, input_shape)
    elif config.ddpm_type == 'VDM2_V':
        model = VDM2V(config, input_shape)    
    elif config.ddpm_type == 'S3VA':
        model = Sigma3VA(config, input_shape)
    elif config.ddpm_type == 'S4V_NT':
        model = Sigma4VNT(config, input_shape)      
    else:
        raise NotImplementedError(f'Model type not implemented: {config.ddpm_type}')
        
    height = input_shape[0]
    width = input_shape[1]
    channels = input_shape[2]
    
    images = jnp.zeros((8, height, width, channels), 'uint8')
    conditioning = jnp.zeros((8,))
    rng1, rng2, rng3, rng4 = jax.random.split(rng, num = 4)
    params = model.init({'params': rng1, 'sample': rng2, 'dropout': rng3, 'encode': rng4}, 
                        images = images,
                        conditioning = conditioning)
    return model, params


