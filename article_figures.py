#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:46:57 2024

@author: bemi


Generate plots for the article.


"""




from src.config_classes.ddpm_config import DDPMConfig
from src.config_classes.enc_dec_config import EncDecConfig
from src.config_classes.nn_config import NNConfig
from src.config_classes.optimizer_config import OptimizerConfig, OptimizerArgs
from src.config_classes.training_config import TrainingConfig


from src.evaluation import display
from src.evaluation.encoder_inspector import EncoderInspector



# Save relevant sample files in [base_folder] and relevant checkpoints in 
# [base_folder]/checkpoints  
# figures will be saved in [save_folder] 
# Checkpoints:
# 2023-09-25_S3VA_fixed_-13_3_5_0_unet_32_unet_mult_res_4_m1_to_1_cifar10_2_128
# 2023-09-05_S3VA_fixed_-13_3_5_0_unet_8_unet_2_m1_to_1_mnist_1_128
#
# Sample files:
# 2023-08-22_S3VA_fixed_-13_3_5_0_unet_8_unet_2_m1_to_1_cifar10_2_128_1400000_samples_256_000.h5
# 2023-09-25_S3VA_fixed_-13_3_5_0_unet_32_unet_mult_res_4_m1_to_1_cifar10_2_128_8000000_samples_256_000.h5
# 2023-09-25_VDM2_V_fixed_-13_3_5_0_unet_32_simple_cifar10_2_128_8000000_samples_256_000.h5

def make_all_article_figures(base_folder: str, save_folder: str) -> None:
    
    make_heatmaps(base_folder, save_folder)
    make_comparison_of_samples(base_folder, save_folder)
    make_encoded_mnist_example(base_folder, save_folder)


def make_heatmaps(base_folder: str, save_folder: str) -> None:
    
    # 2023-09-25_S3VA_fixed_-13_3_5_0_unet_32_unet_mult_res_4_m1_to_1_cifar10_2_128
    model_config = DDPMConfig(
        ddpm_type='S3VA',
      
        vocab_size=256,    
        sample_softmax=False,
        
        n_timesteps = 0,   
        antithetic_time_sampling=True,
        loss_parameterisation = 'eps_hat',
        no_recon_loss = False,
        

        # configurations of the noise schedule
        gamma_type='fixed',
        gamma_min=-13.3,
        gamma_max=5.0,

        # configurations of the score model
        score_model_config = NNConfig(        
            m_type = 'unet',
            
            with_fourier_features=True,
            with_attention=False,
            
            n_embd = 128,
            n_layer = 32,
            num_groups_groupnorm = 32,
            p_dropout = 0.1,
            
            down_conv = False,
            pooling = False,        
            channel_scaling = 'same', # 'same' or 'double'  
            non_id_init = False
            ),
        
        # configurations of the encoder
        encoder_config = EncDecConfig(
            enc_type = 'trainable', # simple, trainable
            vocab_size = 256,
            
            nn_config = NNConfig(        
                m_type = 'unet_mult_res',
                
                with_fourier_features=True,
                with_attention=False,
                
                n_embd = 64,
                n_layer = 4,
                num_groups_groupnorm = 16,
                p_dropout = 0.1,
                
                down_conv = False,
                pooling = True,
                channel_scaling = 'double', # 'same' or 'double'
                non_id_init = False             
                ),
            id_at_zero = False,
            m1_to_1 = True,
            k = 1.0,
            gamma_reg = False,
            end_reg = False,
            end_half_reg = False
            )
      )



    opt_config = OptimizerConfig(    
        name='adamw',
        args=OptimizerArgs(
              b1=0.9,
              b2=0.99,
              eps=1e-8,
              weight_decay=0.01,
          ),
        learning_rate=2e-4,
        use_gradient_clipping = False,
        gradient_clip_norm = 2.0
        )

    train_config = TrainingConfig(
        dataset_name='cifar10_unconditional',
        seed=2,
        num_steps_train=5000,    
        batch_size=128,
        
        steps_per_logging=100,
        steps_per_eval=500,
        steps_per_save=1000,
        step_saves_to_keep = [20000, 100000],
        profile=False,
        )
    #model_name = naming.get_model_name(model_config, train_config, opt_config)

    inspector = EncoderInspector(
        train_config, model_config, opt_config, base_folder, '2023-09-25', 8000000)

    inspector.set_examples('eval', num_classes = 10, num_examples = 10, 
                           eval_batch_idx = 2)
    

    inspector.get_article_heatmaps_x_t_changes(
        save_folder, # save_to_folder
        show_late_t = False,
        ts_for_table = [4, 6, 8, 10],
        example_idxs = [1, 3, 4, 7],
        move_to_0_to_1_from_m2_2 = True)

    inspector.get_article_heatmaps_x_t_changes(
        save_folder, # save_to_folder
        show_late_t = False,
        ts_for_table = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        example_idxs = [1, 3, 4, 7, 0, 2, 5, 6, 8, 9],
        move_to_0_to_1_from_m2_2 = True)
    

def make_comparison_of_samples(
        base_folder: str, save_folder: str) -> None:
    
    sample_files = [
       '2023-08-22_S3VA_fixed_-13_3_5_0_unet_8_unet_2_m1_to_1_cifar10_2_128_1400000_samples_256_000.h5',
       '2023-09-25_S3VA_fixed_-13_3_5_0_unet_32_unet_mult_res_4_m1_to_1_cifar10_2_128_8000000_samples_256_000.h5',
       '2023-09-25_VDM2_V_fixed_-13_3_5_0_unet_32_simple_cifar10_2_128_8000000_samples_256_000.h5'
    ]
        
    display.display_comparison_samples_from_h5(
        base_folder, sample_files, dataset = 'cifar10', save_dir = save_folder)
    

def make_encoded_mnist_example(base_folder: str, save_folder: str) -> None:
    
    # 2023-09-05_S3VA_fixed_-13_3_5_0_unet_8_unet_2_m1_to_1_mnist_1_128
    model_config = DDPMConfig(
        ddpm_type='S3VA',
      
        vocab_size=256,    
        sample_softmax=False,
        
        n_timesteps = 0,   
        antithetic_time_sampling=True,
        loss_parameterisation = 'eps_hat',
        no_recon_loss = False,
        

        # configurations of the noise schedule
        gamma_type='fixed',  # learnable_scalar /  fixed
        gamma_min=-13.3,
        gamma_max=5.0,

        # configurations of the score model
        score_model_config = NNConfig(        
            m_type = 'unet',
            
            with_fourier_features=True,
            with_attention=False,
            
            n_embd = 128,
            n_layer = 8,
            num_groups_groupnorm = 32,
            p_dropout = 0.1,
            
            down_conv = False,
            pooling = False,        
            channel_scaling = 'same', # 'same' or 'double'  
            non_id_init = False
            ),
        
        # configurations of the encoder
        encoder_config = EncDecConfig(
            enc_type = 'trainable', # simple, trainable, trainable_t
            vocab_size = 256,
            
            nn_config = NNConfig(        
                m_type = 'unet',
                
                with_fourier_features=True,
                with_attention=False,
                
                n_embd = 64,
                n_layer = 2,
                num_groups_groupnorm = 16,
                p_dropout = 0.1,
                
                down_conv = False,
                pooling = True,
                channel_scaling = 'double', # 'same' or 'double'
                non_id_init = False             
                ),
            id_at_zero = False,
            m1_to_1 = True,
            k = 1.0,
            gamma_reg = False,
            end_reg = False,
            end_half_reg = False
            )
      )



    opt_config = OptimizerConfig(    
        name='adamw',
        args=OptimizerArgs(
              b1=0.9,
              b2=0.99,
              eps=1e-8,
              weight_decay=0.01,
          ),
        learning_rate=2e-4,
        use_gradient_clipping = False,
        gradient_clip_norm = 2.0
        )

    train_config = TrainingConfig(
        dataset_name='mnist_unconditional',
        seed=1,
        num_steps_train=5000,    
        batch_size=128,
        
        steps_per_logging=100,
        steps_per_eval=500,
        steps_per_save=1000,
        step_saves_to_keep = [20000, 100000],
        profile=False,
        )
    #model_name = naming.get_model_name(model_config, train_config, opt_config)

    inspector = EncoderInspector(
        train_config, model_config, opt_config, base_folder, '2023-09-05', 1400000)

    inspector.set_examples('eval', num_classes = 10, num_examples = 10,
                           eval_batch_idx = 2)


    inspector.get_encoded_for_ts(save_folder, 
                                 ts_for_table = [0.0, 0.70, 0.80, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0],
                                 example_idxs = [0, 3, 6, 9],
                                 move_to_0_to_1_from_m1_1 = True)




# base_folder = 'insert_base_folder_here'
# save_folder = 'insert_base_folder_here'
#
# make_all_article_figures(base_folder, save_folder)




