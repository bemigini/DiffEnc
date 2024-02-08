#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:34:22 2022

@author: bemi


Synthetic data example


Based on: https://github.com/google-research/vdm
https://github.com/google-research/vdm/blob/main/colab/2D_VDM_Example.ipynb


# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""



import numpy as np

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax._src.random import PRNGKey
from jaxtyping import Array

from flax import core
import flax.jax_utils as flax_utils
import optax

from tensorflow.data import Dataset

from typing import Any, Dict
from tqdm import tqdm


from src.basic_ddpm.model import VDM
from src.config_classes.ddpm_config import DDPMConfig
from src.config_classes.optimizer_config import OptimizerConfig, OptimizerArgs
from src.config_classes.training_config import TrainingConfig
from src.data import load_data 
from src.data.synthetic_data import make_8_bit_swirl_data
from src.training.train_state import TrainState





# Data hyper-parameters
N = 1024                 # nr of datapoints

# Model hyper-parameters
init_gamma_0 = -13.3    # initial gamma_0
init_gamma_1 = 5.       # initial gamma_1
hidden_units = 512
T_train = 0                   # nr of timesteps in model; T=0 means continuous-time
vocab_size = 256

# Optimization hyper-parameters
learning_rate = 3e-3
num_train_steps = 20000   # nr of training steps

rng = jax.random.PRNGKey(seed=0)
np.random.seed(0)


data = make_8_bit_swirl_data(N)
data_reshaped = jnp.expand_dims(data, axis = (1, 2))



def plot(data, color='blue'):    
    plt.scatter(data[:,0], data[:,1], alpha=0.1, c=color)
    


plot(data)


data_mean = data.mean(axis=0)
data_std = data.std(axis=0)




def get_model_and_params(config: DDPMConfig, rng: PRNGKey):
    if config.ddpm_type == 'VDM':
        model = VDM(config)
    else: 
        raise NotImplementedError(f'Model type not implemented: {config.ddpm_type}')
    
    image_shape = config.image_shape
    height = image_shape[0]
    width = image_shape[1]
    
    images = 128*jnp.ones((2, height, width, config.number_of_channels), 'uint8')
    conditioning = jnp.zeros((2,))
    rng1, rng2 = jax.random.split(rng)
    params = model.init({'params': rng1, 'sample': rng2}, 
                        images = images,
                        conditioning = conditioning)
    return model, params


def loss_fn(params, images: Array, conditioning: Array, 
            state: TrainState, rng: PRNGKey, is_train: bool):
    rng, sample_rng = jax.random.split(rng)
    rngs = {'sample': sample_rng }
    if is_train:
        rng, dropout_rng = jax.random.split(rng)
        rngs['dropout'] = dropout_rng
        
    # sample time steps 
    outputs = state.apply_fn(
        variables = {'params': params},
        images = images,
        conditioning = conditioning,
        rngs = rngs,
        deterministic = not is_train)
    
    rescale_to_bpd = 1./(np.prod(images.shape[1:]) * np.log(2.))
    bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd
    bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd
    bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd
    loss = bpd_recon + bpd_latent + bpd_diff
    scalar_dict = {
        "bpd": loss,
        "bpd_latent": bpd_latent,
        "bpd_recon": bpd_recon,
        "bpd_diff": bpd_diff,
        "var0": outputs.var_0,
        "var": outputs.var_1,
    }
    img_dict = {"inputs": images}
    metrics = {"scalars": scalar_dict, "images": img_dict}

    return loss, metrics


def get_optimizer(learning_rate: float):
    return optax.adamw(learning_rate)


def get_train_state(
        model: VDM, 
        params: core.FrozenDict[str, Any]) -> TrainState:
    
    state = TrainState.create(
        apply_fn = model.apply,
        variables = params,
        optax_optimizer = get_optimizer)
    return state


@jax.jit
def train_step(
        rng: PRNGKey, state: TrainState, batch: Dict, learning_rate: float):
    
    rng, rng1 = jax.random.split(rng)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, metrics), grads = grad_fn(
        state.params, batch, state, rng1, is_train = True)
    
    new_state = state.apply_gradients(
        grads = grads, lr = learning_rate, ema_rate = 0.9999)
    
    return new_state, loss, metrics, rng
    
    




# define sampling function
# t_end is integer between 0 and T_sample
def sample_fn(rng, N_sample, T_sample, model, state):
    # sample z_0 from the diffusion model
    rng, rng1 = jax.random.split(rng)
    z = [jax.random.normal(rng1, (N_sample, 1, 1, 2))]
    
    for i in tqdm(range(T_sample)):
        rng, rng1 = jax.random.split(rng)
        _z = state.apply_fn(
            variables={'params': state.params},
            i = i, 
            T = T_sample, 
            z_t = z[-1], 
            conditioning = jnp.zeros((z[-1].shape[0],)), 
            rng = rng1, 
            method = model.sample)
        z.append(_z)
    
    x_sample = state.apply_fn(state.params, z[-1], method=model.generate_x)
    
    return z, x_sample







model_config = DDPMConfig(
    ddpm_type='VDM',
  
    vocab_size=256,
    number_of_channels = 2,
    image_shape = (1,1),
    sample_softmax=False,
    
    antithetic_time_sampling=True,
    with_fourier_features=True,
    with_attention=False,

    # configurations of the noise schedule
    gamma_type='fixed',  # learnable_scalar / learnable_nnet / fixed
    gamma_min=-13.3,
    gamma_max=5.,

    # configurations of the score model
    sm_type = 'small',
    sm_n_timesteps=0,
    sm_n_embd=256,
    sm_n_layer=32,
    sm_last_layer_dim = 2,
    sm_pdrop=0.1,
  )


opt_config = OptimizerConfig(    
    name='adamw',
    args=OptimizerArgs(
          b1=0.9,
          b2=0.99,
          eps=1e-8,
          weight_decay=0.01,
      ),
    learning_rate=3e-3,
    lr_decay=False,
    ema_rate=0.9999,
    )

train_config = TrainingConfig(
    dataset_name='synthetic',
    seed=1,
    substeps=1000,
    num_steps_lr_warmup=100,
    num_steps_train=5000,
    num_steps_eval=100,
    batch_size_train=512,
    batch_size_eval=512,
    steps_per_logging=1000,
    steps_per_eval=2000,
    steps_per_save=2000,
    profile=False,
    )


def get_lr_schedule():
    learning_rate = opt_config.learning_rate
    config_train = train_config
    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=config_train.num_steps_lr_warmup
    )

    if opt_config.lr_decay:
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








seed = train_config.seed 
rng = jax.random.PRNGKey(seed)

train_ds, eval_ds = load_data.load_dataset(
    train_config.dataset_name, train_config.batch_size_train)
train_metrics = []

# Initialize model
rng, model_rng = jax.random.split(rng)
model, params = get_model_and_params(model_config, model_rng)

# Create train state
train_state = TrainState.create(
    apply_fn = model.apply,
    variables = params,
    optax_optimizer = get_optimizer)
lr_schedule = get_lr_schedule()


# Initialize train steps
rng, train_rng = jax.random.split(rng)




def train_step_base(
        state: TrainState, 
        batch: Dataset):
                
    # rng1 = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    rng1 = jax.random.fold_in(train_rng, state.step[0])
    
    learning_rate = lr_schedule(state.step)
    images, conditioning = batch
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, metrics), grads = grad_fn(
        state.params, images, conditioning, state, rng1, is_train = True)
    
    ema_rate = opt_config.ema_rate
    new_state = state.apply_gradients(
        grads = grads, lr = learning_rate, ema_rate = ema_rate)
    
    return new_state, metrics



state = train_state

iter_batches = iter(train_ds)

for i in range(train_config.num_steps_train):
    jnp_batch = jax.tree_util.tree_map(jnp.asarray, next(iter_batches))
    
    state, train_metrics = jax.lax.scan(train_step_base, state, jnp_batch)
        
    if i % train_config.steps_per_logging == 0 or i == train_config.num_steps_train - 1:
        metrics = train_metrics['scalars']
        print('appending metrics')
        train_metrics.append(metrics)








model, params = get_model_and_params(model_config, rng)



learning_rate = 3e-3

"""
model = VDM(config)
inputs = {"images": jnp.zeros((2, 1, 1, 2), "uint8")}
inputs['conditioning'] = jnp.zeros((2,))
rng1, rng2 = jax.random.split(rng)
rngs = {'params': rng1, 'sample': rng2}
params = model.init(rngs = rngs, **inputs)
"""

train_state = get_train_state(model, params)
state = train_state


# initialize optimizer
#optimizer = optax.adamw(learning_rate)
#optim_state = optimizer.init(params)
train_inputs = {"images": data_reshaped}
train_inputs['conditioning'] = jnp.zeros((data_reshaped.shape[0],))


num_train_steps = 5000   # nr of training steps    

# training loop (should take ~20 mins)
# rng: PRNGKey, state: TrainState, batch: Dict, learning_rate
losses = []
metrics = []
for i in tqdm(range(num_train_steps)):
    state, loss, _metrics, rng = train_step(rng, state, train_inputs, learning_rate)
    losses.append(loss)
    metrics.append(_metrics)



plt.plot(losses)

metrics[0]['scalars']
metrics[500]['scalars'] 
metrics[1000]['scalars'] 
metrics[2000]['scalars']
metrics[4500]['scalars']

g_0 = model.apply(state.params, 0., method = model.get_gamma) # e.g. -13.0481205
g_1 = model.apply(state.params, 1., method = model.get_gamma) # e.g. 5.741289


rng, rng1 = jax.random.split(rng)
z, x_sample = sample_fn(rng1, N_sample=1024, T_sample=200, model = model, state = state)

    

type(z)
type(z[0])

plot(jnp.squeeze(z[0], axis = (1, 2)))

plot(jnp.squeeze(z[100], axis = (1, 2)))

plot(jnp.squeeze(z[110], axis = (1, 2)))

plot(jnp.squeeze(z[120], axis = (1, 2)))

plot(jnp.squeeze(z[150], axis = (1, 2)))

plot(jnp.squeeze(z[199], axis = (1, 2)))


gen_x = jnp.squeeze(x_sample, axis = (1, 2))

plt.scatter(gen_x[:,0],gen_x[:,1], alpha=0.1)





from absl import logging

logging.set_verbosity(logging.DEBUG)
seed = 1
bla = jax.random.PRNGKey(seed)



