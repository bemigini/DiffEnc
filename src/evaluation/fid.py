#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 08:13:21 2023

@author: bemi



Computing statistics for FID score



"""



from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

import logging

import jax.numpy as jnp
import numpy as np

from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

from skimage.transform import resize


from typing import Tuple



def compute_fid_statistics(iter_batches, input_shape: Tuple[int, int, int]):
    
    model = InceptionV3(
        include_top=False, pooling='avg', input_shape=input_shape)
    
    i = 0
    act = []
    for batch in iter_batches:
        logging.info(f'Batch {i}')
        images, conditioning = batch
        images = np.array(images)
        
        if images[0].shape != input_shape:
            images = scale_images(images, input_shape)
        
        images = preprocess_input(images)
        
        pred = model.predict(images)
        
        act.append(pred)
        i += 1
        
    act = jnp.concatenate(act, axis=0)

    mu = jnp.mean(act, axis=0)
    sigma = jnp.cov(act, rowvar=False)
    return mu, sigma
        

def scale_images(images, new_shape: Tuple[int, int, int]):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.array(images_list)


 
# calculate frechet inception distance from means and variances
def calculate_fid_score(mu1, sigma1, mu2, sigma2):    
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid















