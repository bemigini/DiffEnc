#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:05:06 2023

@author: bemi


For evaluation of an already trained model.


"""



from flax.training.common_utils import stack_forest

from flax.training.train_state import TrainState


import jax
import jax.numpy as jnp
import jax.tree_util 
from jaxtyping import Array

import logging

import os

import numpy as np

from tensorflow.data import Dataset

from typing import Tuple

from tqdm import tqdm

from src.config_classes.ddpm_config import DDPMConfig
from src.config_classes.optimizer_config import OptimizerConfig
from src.config_classes.training_config import TrainingConfig

from src.data import load_data 
from src.data.sample_dataset import SampleDatasetGenerator

from src.evaluation.fid import compute_fid_statistics, calculate_fid_score
from src.file_handling import naming, save_load_json
from src.model_handling import ModelHandler
from src.training.training_pipeline import loss_fn



class Evaluator(ModelHandler):
        
    def __init__(self, 
                 train_config: TrainingConfig, 
                 model_config: DDPMConfig,
                 optimizer_config: OptimizerConfig,
                 output_folder: str,
                 date_str: str,
                 step: int,
                 imagenet_folder: str = ''):
        
        if date_str == '':
            raise ValueError('A date string must be chosen.')
        
        super().__init__(train_config, model_config, optimizer_config)
        
        self.output_folder = output_folder
        checkpoint_dir = os.path.join(output_folder, 'checkpoints')
        metrics_dir = os.path.join(output_folder, 'metrics')
        keep_dir = os.path.join(output_folder, 'keep')
        eval_dir = os.path.join(output_folder, 'evaluation')
        if not os.path.exists(checkpoint_dir):
           os.mkdir(checkpoint_dir) 
        if not os.path.exists(metrics_dir):
           os.mkdir(metrics_dir) 
        if not os.path.exists(keep_dir):
           os.mkdir(keep_dir) 
        if not os.path.exists(eval_dir):
           os.mkdir(eval_dir) 
          
        self.imagenet_folder = imagenet_folder
        
        self.train_ds, self.eval_ds, self.condition_classes = load_data.load_dataset(
            self.train_config, imagenet_folder)
        
        self.eval_steps = load_data.get_eval_steps(
            self.train_config.dataset_name, self.train_config.batch_size)
        
        self.rng, train_rng = jax.random.split(self.rng)
        self.train_rng = train_rng
        self.rng, eval_rng = jax.random.split(self.rng)
        self.eval_rng = eval_rng
        
        prefix = naming.get_file_prefix(self.model_config, self.train_config, self.optimizer_config, date_str)
        restored_state = self.load_checkpoint(output_folder, prefix, step)
        self.train_state = restored_state
        self.prefix = prefix 
        
        self.date_str = date_str 
        self.step = step 
        
        
    def eval_step(self, state: TrainState, batch: Dataset, eval_step=0,
                  ts: Array = jnp.array([])):
        rng = jax.random.fold_in(self.eval_rng, eval_step)
        
        images, conditioning = batch
        if self.condition_classes == 0:
            conditioning = jnp.zeros(conditioning.shape)
        
        _, metrics = loss_fn(state.ema_params, images, conditioning, state, 
                             rng=rng, is_train=False,
                             ts = ts)
            
        return metrics
        
    
    def overwrite_data(self, train_config: TrainingConfig, imagenet_folder):
        
        self.imagenet_folder = imagenet_folder
        
        self.train_ds, self.eval_ds, self.condition_classes = load_data.load_dataset(
            train_config, imagenet_folder)
        
        self.eval_steps = load_data.get_eval_steps(
            train_config.dataset_name, train_config.batch_size)
        
    
    
    def get_bpd_evaluation_metrics(self): 
        jitted_eval_step = jax.jit(self.eval_step)
        
        state = self.train_state
        
        iter_eval = iter(self.eval_ds)
        
        if self.imagenet_folder != '':
            iter_eval = iter(next(iter_eval))
            
        eval_metrics = []
        for eval_step in tqdm(range(self.eval_steps)):
            eval_batch = jax.tree_util.tree_map(jnp.asarray, next(iter_eval))
            metrics = jitted_eval_step(state, eval_batch)
            eval_metrics.append(metrics['scalars'])
            
        mean_eval_metrics = stack_forest(eval_metrics)
        mean_eval_metrics = jax.tree_map(jnp.mean, mean_eval_metrics)
        steps = int(state.step)
        eval_metrics_dict = {steps: {'eval': mean_eval_metrics}}
        
        return eval_metrics_dict
    
    
    def get_bpd_evaluation_per_t_metrics(self, step_size: float = 0.1): 
        to_val = 1. + step_size
        per_t = jnp.arange(0, to_val, step = step_size)
        
        iter_eval = iter(self.eval_ds)
        eval_batch = jax.tree_util.tree_map(jnp.asarray, next(iter_eval))
        images, conditioning = eval_batch
        batch_size = images.shape[0]
        
        state = self.train_state
        
        steps = int(state.step)
        eval_metrics_dict = {steps: {}}
        
        jitted_eval_step = jax.jit(self.eval_step)
        
        for t in tqdm(per_t):
            ts = jnp.ones(batch_size) * t             
                        
            iter_eval = iter(self.eval_ds)
            eval_metrics = []
            for eval_step in range(self.eval_steps):
                eval_batch = jax.tree_util.tree_map(jnp.asarray, next(iter_eval))
                metrics = jitted_eval_step(state, eval_batch, ts = ts)
                eval_metrics.append(metrics['scalars'])
                
            mean_eval_metrics = stack_forest(eval_metrics)
            mean_eval_metrics = jax.tree_map(jnp.mean, mean_eval_metrics)
            t_key = f'{t:.2f}'
            eval_metrics_dict[steps][t_key] = {'eval': mean_eval_metrics}
            
        
        return eval_metrics_dict
    
    
    def save_eval_to_json(self, 
                          eval_dir: str = '',                           
                          per_t: float = 0.1):
        
        if per_t is not None and per_t > 0:
            eval_metrics_dict = self.get_bpd_evaluation_per_t_metrics(per_t)
            per_t_suffix = '_per_t'
        else:
            eval_metrics_dict = self.get_bpd_evaluation_metrics()
            per_t_suffix = ''
                
        if eval_dir == '':
            eval_dir = os.path.join(self.output_folder, 'evaluation')
        
        eval_path = os.path.join(eval_dir, f'{self.prefix}evaluation_metrics{per_t_suffix}.json')
        save_load_json.save_as_json(eval_metrics_dict, eval_path)
        
    
    def get_eval_fid_statistics(self, 
                                out_dir: str = '', 
                                input_shape: Tuple[int, int, int] = (256,256,3),
                                wrt_train: bool = False) -> None:
        if out_dir == '':
            out_dir = os.path.join(self.output_folder, 'evaluation')
            
        wrt_train_eval = 'train' if wrt_train else 'eval'
                
        save_path = os.path.join(out_dir, f'fid_stats_{wrt_train_eval}_{self.train_config.dataset_name}.npz')
        self.eval_stats_path = save_path
        
        if os.path.isfile(save_path):
            logging.info(f'Pre-computed fid statistics already exist at {save_path}')
            return
        logging.info('Computing eval FID statistics')
        if wrt_train:
            logging.info('With respect to train data')
            train_as_eval_ds = load_data.load_train_cifar10_as_eval(
                self.train_config.batch_size)
            iter_batches = iter(train_as_eval_ds)
        else:
            logging.info('With respect to test data')
            iter_batches = iter(self.eval_ds)
                   
        
        mu, sigma = compute_fid_statistics(iter_batches, input_shape)        
        
        np.savez(save_path, mu=mu, sigma=sigma)
        
        logging.info(f'Saved pre-computed statistics at: {save_path}')
        
    
    def get_samples_fid_statistics(self,
                                sample_dir: str = '', 
                                out_dir: str = '', 
                                batch_size: int = 100,
                                input_shape: Tuple[int, int, int] = (256,256,3) ) -> None:
        if out_dir == '':
            out_dir = os.path.join(self.output_folder, 'evaluation')
                
        save_path = os.path.join(out_dir, f'fid_stats_samples_{self.prefix}{self.step}.npz')
        self.sample_stats_path = save_path
        
        if os.path.isfile(save_path):
            logging.info(f'Pre-computed fid statistics already exist at {save_path}')
            return
        
        # Get samples
        sample_file_prefix = naming.get_sample_file_prefix(
            self.prefix, 
            self.step, 
            batch_size)
        logging.info('Getting sample dataset')
        
        sampleGen = SampleDatasetGenerator(sample_file_prefix, sample_dir)
        iter_batches = iter(sampleGen)
        
        
        logging.info('Computing samples FID statistics')                
        mu, sigma = compute_fid_statistics(iter_batches, input_shape)        
        
        np.savez(save_path, mu=mu, sigma=sigma)
        
        logging.info(f'Saved pre-computed statistics at: {save_path}')
        
        
    def get_fid_score(self, wrt_train: bool = False) -> None:
        
        if self.sample_stats_path is None or self.sample_stats_path == '' or self.eval_stats_path is None or self.eval_stats_path == '':
            raise ValueError(f'No FID statistics found: eval: {self.eval_stats_path}, samples: {self.sample_stats_path}')
        
        eval_stats = np.load(self.eval_stats_path)
        eval_mu = eval_stats['mu']
        eval_sigma = eval_stats['sigma']
        sample_stats = np.load(self.sample_stats_path)
        sample_mu = sample_stats['mu']
        sample_sigma = sample_stats['sigma']
        
        fid_score = calculate_fid_score(eval_mu, eval_sigma, sample_mu, sample_sigma)
        score_dict = {'FID': fid_score}
        
        logging.info(f'FID score: {fid_score}')
        
        if wrt_train:
            wrt_train_str = 'wrt_train_'
        else:
            wrt_train_str = ''
        
        eval_dir = os.path.join(self.output_folder, 'evaluation')
        fid_file = f'fid_score_{wrt_train_str}{self.prefix}{self.step}.json'
        fid_path = os.path.join(eval_dir, fid_file)
        
        save_load_json.save_as_json(score_dict, fid_path)
        
        
    def get_fid_score_test_vs_train_CIFAR10(self) -> None:
        
        self.get_eval_fid_statistics('', (256,256,3), False)
        
        fid_stat_path_test = self.eval_stats_path
        
        self.get_eval_fid_statistics('', (256,256,3), True)
        
        fid_stat_path_train = self.eval_stats_path
        
        
        eval_stats = np.load(fid_stat_path_train)
        eval_mu = eval_stats['mu']
        eval_sigma = eval_stats['sigma']
        sample_stats = np.load(fid_stat_path_test)
        sample_mu = sample_stats['mu']
        sample_sigma = sample_stats['sigma']
        
        fid_score = calculate_fid_score(eval_mu, eval_sigma, sample_mu, sample_sigma)
        score_dict = {'FID': fid_score}
        
        logging.info(f'FID score: {fid_score}')
                
        eval_dir = os.path.join(self.output_folder, 'evaluation')
        fid_file = 'fid_score_CIFAR-10_test_vs_train.json'
        fid_path = os.path.join(eval_dir, fid_file)
        
        save_load_json.save_as_json(score_dict, fid_path)
        
    
    def get_fid_score_train_vs_test_CIFAR10(self) -> None:
        
        self.get_eval_fid_statistics('', (256,256,3), False)
        
        fid_stat_path_test = self.eval_stats_path
        
        self.get_eval_fid_statistics('', (256,256,3), True)
        
        fid_stat_path_train = self.eval_stats_path
        
        
        eval_stats = np.load(fid_stat_path_test)
        eval_mu = eval_stats['mu']
        eval_sigma = eval_stats['sigma']
        sample_stats = np.load(fid_stat_path_train)
        sample_mu = sample_stats['mu']
        sample_sigma = sample_stats['sigma']
        
        fid_score = calculate_fid_score(eval_mu, eval_sigma, sample_mu, sample_sigma)
        score_dict = {'FID': fid_score}
        
        logging.info(f'FID score: {fid_score}')
                
        eval_dir = os.path.join(self.output_folder, 'evaluation')
        fid_file = 'fid_score_CIFAR-10_train_vs_test.json'
        fid_path = os.path.join(eval_dir, fid_file)
        
        save_load_json.save_as_json(score_dict, fid_path)
        
    
        



