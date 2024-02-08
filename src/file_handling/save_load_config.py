#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 08:24:08 2023

@author: bemi


For saving and loading configs


"""


import os

from typing import Dict

from src.config_classes.ddpm_config import DDPMConfig
from src.config_classes.enc_dec_config import EncDecConfig
from src.config_classes.nn_config import NNConfig
from src.config_classes.optimizer_config import OptimizerConfig, OptimizerArgs
from src.config_classes.training_config import TrainingConfig
from src.file_handling import save_load_json as sljson
from src.file_handling import naming


def save_ddpm_config(config: DDPMConfig, save_to_folder:str) -> None:
    
    suffix = naming.get_model_config_suffix(config)
    
    file_name = f'config_model_{suffix}.json'
    save_config(config, save_to_folder, file_name)
    

def save_optimizer_config(config: OptimizerConfig, save_to_folder:str) -> None:
    
    learning_rate_str = str(config.learning_rate).split('.')[1]
    
    file_name = f'config_opt_{config.name}_{learning_rate_str}.json'
    
    save_config(config, save_to_folder, file_name)
    

def save_train_config(config: TrainingConfig, save_to_folder:str) -> None:
    
    suffix = naming.get_train_config_suffix(config)
    
    file_name = f'config_train_{suffix}.json'
    
    save_config(config, save_to_folder, file_name)


def save_config(config, save_to_folder: str, file_name: str) -> None:
    save_to_path = os.path.join(save_to_folder, file_name)
    sljson.save_as_json(config, save_to_path)


def load_ddpm_config(load_from_file: str, load_from_folder: str) -> DDPMConfig:
    load_from_path = os.path.join(load_from_folder, load_from_file)
    json_config = sljson.load_json(load_from_path)
        
    json_config['score_model_config'] = nn_config_from_dict(json_config['score_model_config'])
    encdec_config_dict = json_config['encoder_config']
    json_config['encoder_config'] = enc_dec_config_from_dict(encdec_config_dict)
        
    return DDPMConfig(**json_config)
    

def enc_dec_config_from_dict(encdec_config_dict: Dict) -> EncDecConfig:
    
    if encdec_config_dict['nn_config'] is not None:
        encdec_config_dict['nn_config'] = nn_config_from_dict(encdec_config_dict['nn_config'])
    
    return EncDecConfig(**encdec_config_dict)


def nn_config_from_dict(nn_config: Dict):
    if not nn_config['kernel_init']:
        del nn_config['kernel_init']
    
    return NNConfig(**nn_config)


def load_optimizer_config(load_from_file: str, load_from_folder: str) -> OptimizerConfig:
    load_from_path = os.path.join(load_from_folder, load_from_file)
    json_config = sljson.load_json(load_from_path)
    
    json_config['args'] = OptimizerArgs(**json_config['args'])
    
    return OptimizerConfig(**json_config)


def load_train_config(
        load_from_file: str, 
        load_from_folder: str,
        overwrites: Dict) -> TrainingConfig:
    load_from_path = os.path.join(load_from_folder, load_from_file)
    json_config = sljson.load_json(load_from_path)
    
    for key in overwrites:
        json_config[key] = overwrites[key]
    
    return TrainingConfig(**json_config)







