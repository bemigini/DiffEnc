#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:46:08 2023

@author: bemi


For making consistent filenames


"""


from src.config_classes.ddpm_config import DDPMConfig
from src.config_classes.enc_dec_config import EncDecConfig
from src.config_classes.nn_config import NNConfig
from src.config_classes.optimizer_config import OptimizerConfig
from src.config_classes.training_config import TrainingConfig


def get_model_config_suffix(config: DDPMConfig) -> str:
    m_type = config.ddpm_type
    
    g_type = config.gamma_type
    gamma_min = str(config.gamma_min).replace('.', '_')
    gamma_max = str(config.gamma_max).replace('.', '_')
    gamma_range = f'{gamma_min}_{gamma_max}'
    if config.loss_parameterisation != 'eps_hat':
        loss_p = f'_{config.loss_parameterisation}'
    else:
        loss_p = ''
    
    if config.no_recon_loss:
        no_recon = '_NO_RECON'
    else:
        no_recon = ''
    
    score_model_config = config.score_model_config
    model_info = get_nn_config_suffix(score_model_config)
    
    encoder_config = config.encoder_config
    encoder_info = get_enc_config_suffix(encoder_config)
    
    suffix = f'{m_type}_{g_type}_{gamma_range}{loss_p}{no_recon}_{model_info}_{encoder_info}'
    
    return suffix
    

def get_enc_config_suffix(config: EncDecConfig) -> str:
    
    encoder_type = config.enc_type
    
    if 'trainable' in encoder_type:
        model_info = f'{get_nn_config_suffix(config.nn_config)}'
        
        if config.nn_config.down_conv:
            model_info = model_info + '_downconv'
        #if config.nn_config.pooling:
        #    model_info = model_info + '_maxpool'
        if config.nn_config.non_id_init:
            model_info = model_info + '_non_id_init'
        
        
        if config.gamma_reg:
            model_info = model_info + '_gamma_reg'
        if config.end_reg:
            model_info = model_info + '_end_reg'
        if config.end_half_reg:
            model_info = model_info + '_end_half_reg'
        if config.id_at_zero:
            model_info = model_info + '_id_at_zero'
        if config.m1_to_1:
            if config.k == 1:
                k_str = '1'
            else:
                k_str = str(config.k).replace('.', '_')
            model_info = model_info + f'_m{k_str}_to_{k_str}'
        if config.init_var > 0:
            init_var_str = str(config.init_var).replace('.', '_')
            model_info = model_info + f'_initvar{init_var_str}'
        if config.min_var > 0:
            min_var_str = str(config.min_var).replace('.', '_')
            model_info = model_info + f'_minvar{min_var_str}'
        if config.use_x_var:
            model_info = model_info + '_xvar'
            
    else:
        model_info = ''
    
    encoder_type_str = f'{encoder_type}' if encoder_type != 'trainable' else ''
    
    suffix = f'{encoder_type_str}{model_info}'
    
    return suffix


def get_nn_config_suffix(config: NNConfig) -> str:
    
    m_type = config.m_type
    n_layers = config.n_layer
        
    suffix = f'{m_type}_{n_layers}'
    
    return suffix


def get_train_config_suffix(config: TrainingConfig) -> str:
    
    dataset_str = config.dataset_name.replace('_unconditional', '')
    suffix = f'{dataset_str}_{config.seed}_{config.batch_size}'
    
    return suffix


def get_opt_config_suffix(config: OptimizerConfig) -> str:
    suffix = ''
    
    if config.use_gradient_clipping:
        suffix = f'_clip_{str(config.gradient_clip_norm).replace(".", "_")}'
    
    return suffix


def get_model_name(ddpm_config: DDPMConfig, 
                   train_config: TrainingConfig,
                   opt_config: OptimizerConfig) -> str:
    model_config_suffix = get_model_config_suffix(ddpm_config)
    train_config_suffix = get_train_config_suffix(train_config)
    opt_config_suffix = get_opt_config_suffix(opt_config)
    model_name = f'{model_config_suffix}_{train_config_suffix}{opt_config_suffix}'
    
    return model_name


def get_file_prefix(
        model_config: DDPMConfig, 
        train_config: TrainingConfig,
        opt_config: OptimizerConfig,
        date_str: str) -> str:
    model_name = get_model_name(model_config, train_config, opt_config)
    prefix = f'{date_str}_{model_name}_'
    
    return prefix


def get_file_prefix_from_name(
        model_name: str,
        date_str: str) -> str:   
    prefix = f'{date_str}_{model_name}_'
    
    return prefix


def get_sample_file_prefix(
        file_prefix: str,
        step: int,
        batch_size: int) -> str:
    prefix = f'{file_prefix}{step}_samples_{batch_size}'
    
    return prefix


def get_samples_h5_dataset_name():
    return 'samples'



