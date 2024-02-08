#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:51:26 2022

@author: bemi


Diffusion Model Config 


"""



import flax

from src.config_classes.enc_dec_config import EncDecConfig
from src.config_classes.nn_config import NNConfig


@flax.struct.dataclass
class DDPMConfig:
    ddpm_type: str # 'VDM2', 'VDM2_V', 'S3VA', 'S4V_NT'
        
    vocab_size: int    
    sample_softmax: bool
    
    n_timesteps: int
    antithetic_time_sampling: bool
    
    loss_parameterisation: str # 'x_hat' or 'eps_hat' 
    no_recon_loss: bool # To turn off reconstruction loss
    
    # configurations of the noise schedule
    gamma_type: str # fixed / train_scalar
    gamma_min: float
    gamma_max: float
    
    # configurations of the score model
    score_model_config: NNConfig
    
    
    # configurations of the encoder
    encoder_config: EncDecConfig



