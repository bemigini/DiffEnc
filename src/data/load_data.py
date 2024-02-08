#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:28:47 2022

@author: bemi


Loading dataset


"""



import jax.numpy as jnp

import math
from matplotlib import pyplot

from numpy.typing import NDArray

import tensorflow as tf
from tensorflow.data import Dataset
import tensorflow_datasets as tfds

from typing import Tuple


from src.config_classes.training_config import TrainingConfig
from src.data.synthetic_data import make_8_bit_swirl_data
from src.data.ImageNet_dataset import ImageNetDatasetGenerator




def get_zipped_dataset(X: NDArray, y: NDArray):
    ds_X = tf.data.Dataset.from_tensor_slices(X)
    ds_y = tf.data.Dataset.from_tensor_slices(y)
    
    return tf.data.Dataset.zip((ds_X, ds_y))


def show_first_images(images_pixels: NDArray):
    for i in range(9):  
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(images_pixels[i], cmap=pyplot.get_cmap('gray'))
        pyplot.show()


def load_mnist_conditional() -> Tuple[Dataset, Dataset]:    
    train_data = tfds.load('mnist', split='train', 
                           as_supervised=True,
                           shuffle_files=False)
    test_data = tfds.load('mnist', split='test', as_supervised=True)
    
    return train_data, test_data


def load_synthetic_data_spiral() -> Dataset:        
    N = 1024 # Number of data points
    seed = 1
    data = make_8_bit_swirl_data(N, seed)
    data_reshaped = jnp.expand_dims(data, axis = (1, 2))
    unconditional = jnp.zeros((data_reshaped.shape[0],))
    
    return get_zipped_dataset(data_reshaped, unconditional)


def load_cifar10_conditional() -> Tuple[Dataset, Dataset]:
    train = tfds.load('cifar10', split='train', 
                      as_supervised=True,
                      shuffle_files=False)
    test = tfds.load('cifar10', split='test', as_supervised=True)
    
    return train, test



def get_input_shape(dataset_name: str) -> Tuple[int, int, int]:
    
    dataset_name_root = dataset_name.split('_')[0]    
    
    if dataset_name_root == 'synthetic':        
        input_shape = (1, 1, 2)
    elif dataset_name_root == 'mnist':
        input_shape = (28, 28, 1)   
    elif dataset_name_root == 'cifar10':        
        input_shape = (32, 32, 3)
    elif dataset_name_root == 'imagenet32':
        input_shape = (32, 32, 3)    
    else:
        raise NotImplementedError(f'Dataset not implemented: {dataset_name}')
    
    return input_shape


def get_eval_steps(dataset_name: str, batch_size: int) -> int:
    
    dataset_name_root = dataset_name.split('_')[0]    
    
    if dataset_name_root == 'synthetic':        
        test_size = 1024
    elif dataset_name_root == 'mnist':
        test_size = 10000
    elif dataset_name_root == 'cifar10':
        if 'small_one_class' in dataset_name:
            test_size = 1000
        else:
            test_size = 10000
    elif dataset_name_root == 'imagenet32':
        test_size = 50000
    else:
        raise NotImplementedError(f'Dataset not implemented: {dataset_name}')
    
    return math.floor(test_size/batch_size)


def get_repeats(steps: int, batch_size: int, dataset_size: int):
    repeats = math.ceil((steps * batch_size) / dataset_size)
    return repeats


def load_dataset(train_config: TrainingConfig, imagenet_folder: str = '') -> Tuple[Dataset, Dataset, int]:
    dataset_name = train_config.dataset_name
    batch_size = train_config.batch_size
    train_steps = train_config.num_steps_train
    seed = train_config.seed
    
    if dataset_name == 'synthetic':
        train_ds, eval_ds = load_synthetic_data_spiral(), load_synthetic_data_spiral()
        eval_ds = eval_ds.repeat(2)
        condition_classes = 0
        repeats = get_repeats(train_steps, batch_size, 1024)
    elif dataset_name == 'mnist_conditional':
        train_ds, eval_ds = load_mnist_conditional()
        condition_classes = 10
        repeats = get_repeats(train_steps, batch_size, 60000)
    elif dataset_name == 'mnist_unconditional':
        train_ds, eval_ds = load_mnist_conditional()
        condition_classes = 0
        repeats = get_repeats(train_steps, batch_size, 60000)
    elif dataset_name == 'cifar10_conditional':
        train_ds, eval_ds = load_cifar10_conditional()
        condition_classes = 10
        repeats = get_repeats(train_steps, batch_size, 50000)
    elif dataset_name == 'cifar10_unconditional':
        train_ds, eval_ds = load_cifar10_conditional()
        condition_classes = 0
        repeats = get_repeats(train_steps, batch_size, 50000)    
    elif dataset_name == 'imagenet32_unconditional':
        train_ds = ImageNetDatasetGenerator(
            imagenet_folder = imagenet_folder, img_size = 32, 
            is_train = True, seed = seed, train_steps = train_steps, 
            batch_size = batch_size)
        condition_classes = 0
        eval_ds = ImageNetDatasetGenerator(
            imagenet_folder = imagenet_folder, img_size = 32, 
            is_train = False, seed = seed, train_steps = train_steps, 
            batch_size = batch_size)
    elif dataset_name == 'imagenet64_unconditional':
        train_ds = ImageNetDatasetGenerator(
            imagenet_folder = imagenet_folder, img_size = 64, 
            is_train = True, seed = seed, train_steps = train_steps, 
            batch_size = batch_size)
        condition_classes = 0
        eval_ds = ImageNetDatasetGenerator(
            imagenet_folder = imagenet_folder, img_size = 64, 
            is_train = False, seed = seed, train_steps = train_steps, 
            batch_size = batch_size)
    else:
        raise NotImplementedError(f'Dataset not implemented: {dataset_name}')
    
    if 'imagenet' not in dataset_name:    
        train_ds = train_ds.shuffle(
            buffer_size = 1000, seed = seed, reshuffle_each_iteration=True)
        train_ds = train_ds.repeat(repeats)
        
        train_ds = train_ds.batch(batch_size)
        eval_ds = eval_ds.batch(batch_size)
        
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)
        
    train_ds = iter(train_ds)
    
    return train_ds, eval_ds, condition_classes


def load_train_cifar10_as_eval(batch_size: int) -> Tuple[Dataset, int]:
    train_ds = tfds.load('cifar10', split='train', 
                      as_supervised=True,
                      shuffle_files=False)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds






    


