#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 07:39:44 2023

@author: bemi

run.py: Run Script for training denoising diffusion model. Uses GPU by default.


Usage:
    run.py train --output-folder=<file> --config-folder=<file> --c-model=<file> --c-opt=<file> --c-train=<file> --train-steps=<int> [options]
    run.py train --output-folder=<file> --config-folder=<file> --c-model=<file> --c-opt=<file> --c-train=<file> --date-str=<string> --train-steps=<int> [options]
    run.py sample --output-folder=<file> --config-folder=<file> --c-model=<file> --c-opt=<file> --c-train=<file> --date-str=<string> --train-steps=<int> [options]
    run.py evaluate --output-folder=<file> --config-folder=<file> --c-model=<file> --c-opt=<file> --c-train=<file> --date-str=<string> --train-steps=<int> [options]
    run.py evaluate --output-folder=<file> --config-folder=<file> --c-model=<file> --c-opt=<file> --c-train=<file> --date-str=<string> --train-steps=<int> --sample-folder=<file> --fid [options]
    
    
Options:
    -h --help                               show this screen.
    --output-folder=<file>                  folder to save trained model in
    --config-folder=<file>                  folder to load the config files from    
    --c-model=<file>                        file with model config
    --c-opt=<file>                          file with optimizer config
    --c-train=<file>                        file with training config
    --c-eval=<file>                         file with config for evaluation dataset
    --log=<string>                          log level to use [default: info]    
    --date-str=<string>                     a string with the format YYYY-MM-DD to choose which checkpoint to load. If set with train, training will continue from the checkpoint.
    --num-samples=<int>                     number of samples to draw [default: 25]
    --sample-seed=<int>                     seed for the sample taken [default: 1]
    --sample-batch=<int>                    batch size to use for samples [default: 0]
    --train-seed=<int>                      if set, will overwrite the seed in the training config.
    --train-steps=<int>                     if set for training, will overwrite the number of training steps in the training config. For sampling, will sample checkpoint at this number of steps instead of latest.
    --reverse-noise                         reverse noise from test set examples instead of drawing new samples
    --num-classes=<int>                     number of classes to include in samples [default: 10]
    --per-t-step=<int>                      when evaluating, get loss per t and use this step size
    --fid                                   when evaluating, get FID score
    --fid-wrt-train                         when getting FID score, get it with respect to the train dataset
    --sample-folder=<file>                  when getting FID score, folder with sample images
    --use-T=<int>                           number of steps to use for sampling. Default of 0 means use the number of steps from the model config. [default: 0]
    --imagenet-folder=<file>                folder where ImageNet datasets are located    
    --dt-percentile=<int>                   percentile to use for dynamic thresholding. Set to more than 1, e.g. 75, to use dynamic thresholding [default: 0]
    --percentile-scale=<int>                percentile to use for scaling when sampling. Set to more than 1, e.g. 75, to use scaling when sampling [default: 0]
    --init-var-bound=<int>                  will be divided by 1000. Bound on initial variance when sampling [default: 0]
    
"""




from docopt import docopt

from jax import config
config.update("jax_enable_x64", False) # True if we want to use x64

import logging


from typing import Dict

from src.evaluation.evaluator import Evaluator
from src.evaluation.sampler import Sampler
from src.file_handling import save_load_config
from src.training import training_pipeline




def train(args: Dict):
    
    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    config_folder = args['--config-folder'] if args['--config-folder'] else '.'
    c_model_file = args['--c-model'] if args['--c-model'] else ''
    c_opt_file = args['--c-opt'] if args['--c-opt'] else ''
    c_train_file = args['--c-train'] if args['--c-train'] else ''
    imagenet_folder = args['--imagenet-folder'] if args['--imagenet-folder'] else ''
    
    
    if c_model_file == '' or c_opt_file == '' or c_train_file == '':
        raise ValueError(f'All configs must be given. Given configs: model: {c_model_file}, optimizer: {c_opt_file}, train: {c_train_file}')
    
    
    date_str = args['--date-str'] if args['--date-str'] else ''
    
    overwrites = {}
    if args['--train-seed']:
        seed = int(args['--train-seed']) 
        overwrites['seed'] = seed
    if args['--train-steps']:
        steps = int(args['--train-steps']) 
        overwrites['num_steps_train'] = steps
    
        
    model_config = save_load_config.load_ddpm_config(c_model_file, config_folder)
    opt_config = save_load_config.load_optimizer_config(c_opt_file, config_folder)
    train_config = save_load_config.load_train_config(c_train_file, config_folder, overwrites)
    
    pipeline = training_pipeline.TrainingPipeline(
        train_config, model_config, opt_config, output_folder, date_str, 
        imagenet_folder)
    
    pipeline.train_multi_gpu()
    

def sample(args: Dict):
    
    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    config_folder = args['--config-folder'] if args['--config-folder'] else '.'
    c_model_file = args['--c-model'] if args['--c-model'] else ''
    c_opt_file = args['--c-opt'] if args['--c-opt'] else ''
    c_train_file = args['--c-train'] if args['--c-train'] else ''
    
    if c_model_file == '' or c_opt_file == '' or c_train_file == '':
        raise ValueError(f'All configs must be given. Given configs: model: {c_model_file}, optimizer: {c_opt_file}, train: {c_train_file}')
    
    num_samples = int(args['--num-samples'])
    sample_batch = int(args['--sample-batch'])
    sample_seed = int(args['--sample-seed'])
    date_str = args['--date-str']
    steps = int(args['--train-steps']) if args['--train-steps'] else None
    use_T = int(args['--use-T']) if args['--use-T'] else 0
    
    dt_percentile = float(args['--dt-percentile']) if args['--dt-percentile'] else 0.0
    use_dt = dt_percentile > 0
    percentile_scale = float(args['--percentile-scale']) if args['--percentile-scale'] else 0.0
    init_var_bound = float(args['--init-var-bound']) if args['--init-var-bound'] else 0.0
    init_var_bound = init_var_bound/1000
    
    
    if not date_str:
        raise ValueError('Date string empty')
    
    overwrites = {}
    if args['--train-seed']:
        seed = int(args['--train-seed']) 
        overwrites['seed'] = seed
    
    
    model_config = save_load_config.load_ddpm_config(c_model_file, config_folder)
    opt_config = save_load_config.load_optimizer_config(c_opt_file, config_folder)
    train_config = save_load_config.load_train_config(c_train_file, config_folder, overwrites = overwrites)
    
    
    sampler = Sampler(train_config, model_config, opt_config)
    
    if sample_batch > 0:
        # idx_to_keep = [] to use defaults
        sampler.save_samples_as_h5_multi(
            num_samples, sample_seed, output_folder, date_str, steps, 
            idx_to_keep = [], batch_size = sample_batch, use_T = use_T,
            use_dynamic_thresholding = use_dt, dt_percentile = dt_percentile, 
            percentile_scale = percentile_scale)
    else:
        sampler.save_samples_as_h5(
            num_samples, sample_seed, output_folder, date_str, steps, 
            use_T = use_T,
            use_dynamic_thresholding = use_dt, dt_percentile = dt_percentile, 
            percentile_scale = percentile_scale, init_var_bound = init_var_bound)
        

def evaluate(args: Dict):
    
    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    config_folder = args['--config-folder'] if args['--config-folder'] else '.'
    c_model_file = args['--c-model'] if args['--c-model'] else ''
    c_opt_file = args['--c-opt'] if args['--c-opt'] else ''
    c_train_file = args['--c-train'] if args['--c-train'] else ''
    c_eval_file = args['--c-eval'] if args['--c-eval'] else ''
    imagenet_folder = args['--imagenet-folder'] if args['--imagenet-folder'] else ''
    
    if c_model_file == '' or c_opt_file == '' or c_train_file == '':
        raise ValueError(f'All configs must be given. Given configs: model: {c_model_file}, optimizer: {c_opt_file}, train: {c_train_file}')
    
    sample_batch = int(args['--sample-batch'])
    date_str = args['--date-str']
    steps = int(args['--train-steps']) if args['--train-steps'] else None
    
    if not date_str:
        raise ValueError('Date string empty')
    
    per_t = float(args['--per-t-step']) if args['--per-t-step'] else None
    fid = True if args['--fid'] else False
    fid_wrt_train = True if args['--fid-wrt-train'] else False
    sample_dir = args['--sample-folder'] if args['--sample-folder'] else ''
        
    overwrites = {}
    if args['--train-seed']:
        seed = int(args['--train-seed']) 
        overwrites['seed'] = seed
    
    
    model_config = save_load_config.load_ddpm_config(c_model_file, config_folder)
    opt_config = save_load_config.load_optimizer_config(c_opt_file, config_folder)
    train_config = save_load_config.load_train_config(c_train_file, config_folder, overwrites = overwrites)
        
    
    evaluator = Evaluator(train_config, model_config, opt_config,
                      output_folder, date_str, steps, imagenet_folder)

    if fid:    
        # TODO: make possible to choose image size?
        #evaluator.get_fid_score_test_vs_train_CIFAR10()
        evaluator.get_eval_fid_statistics('', (256,256,3), fid_wrt_train)
        evaluator.get_samples_fid_statistics(
            sample_dir, '', 
            batch_size = sample_batch,
            input_shape = (256,256,3))
        
        evaluator.get_fid_score(fid_wrt_train)
    else:
        if c_eval_file != '':
            eval_config = save_load_config.load_train_config(c_eval_file, config_folder, overwrites = overwrites)
            evaluator.overwrite_data(eval_config, imagenet_folder)
            
        evaluator.save_eval_to_json(per_t = per_t)




def main():     
    
    args = docopt(__doc__)
    
    log_level = args['--log'] if args['--log'] else ''
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=numeric_level)   
    
    
    if args['train']:
        train(args)
    elif args['sample']:
        sample(args)
    elif args['evaluate']:
        evaluate(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()


