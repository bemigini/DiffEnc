#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:46:31 2023

@author: bemi


Model using a time-dependent trainable encoder


Based on https://github.com/google-research/vdm/blob/main/model_vdm.py


"""



from flax import linen as nn
import jax
from jax import numpy as jnp

from jaxtyping import Array
from jax._src.random import PRNGKey

from numpy.typing import NDArray

from typing import Tuple

from src.basic_ddpm.data_classes import VDMOutput
from src.basic_ddpm.noise_schedules import NoiseSchedule_FixedLinear, NoiseSchedule_FixedPoly10, NoiseSchedule_Scalar

from src.config_classes.ddpm_config import DDPMConfig
from src.encode_decode.trainable_encoder_sigma3_param import TrainableEncSigma3Param
from src.nn_models.score_unet import ScoreUNet, ScoreUNetAlt


class Sigma3VA(nn.Module):
    ddpm_config: DDPMConfig
    input_shape: Tuple[int, int, int]
    
    def setup(self):
        config = self.ddpm_config
        self.encdec_config = config.encoder_config        
        
        if 'trainable' in self.encdec_config.enc_type:
            self.encdec = TrainableEncSigma3Param(self.encdec_config,
                                                 config.gamma_min,
                                                 config.gamma_max,
                                                 self.input_shape[0])
        else:
            raise NotImplementedError(f'Encoder type not implemented: {self.encdec_config.enc_type}')
       
        
        if config.score_model_config.m_type == 'unet':
            self.score_model = ScoreUNet(config.score_model_config,
                                         config.gamma_min,
                                         config.gamma_max)
        elif config.score_model_config.m_type == 'unet_alt':
            self.score_model = ScoreUNetAlt(config.score_model_config,
                                         config.gamma_min,
                                         config.gamma_max)
        else:
            raise NotImplementedError(f'Score model not implemented: {config.score_model_config.m_type}')
            
        
        if self.ddpm_config.gamma_type == 'fixed':
            self.gamma = NoiseSchedule_FixedLinear(self.ddpm_config)
        elif self.ddpm_config.gamma_type == 'poly10':
            self.gamma = NoiseSchedule_FixedPoly10(self.ddpm_config)
        elif self.ddpm_config.gamma_type == 'train_scalar':
            self.gamma = NoiseSchedule_Scalar(self.ddpm_config)
        else:
            raise NotImplementedError(f'Noise schedule not implemented: {self.ddpm_config.gamma_type}')
        
        
    def __call__(self, 
                images: Array, 
                conditioning: Array, 
                deterministic: bool = True,
                ts: Array = jnp.array([])) -> VDMOutput:
        
        g_0, g_1 = self.gamma(0.), self.gamma(1.)
        # Variance at time 0 and 1, sigma_0**2, sigma_1**2
        var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
        x = images 
        init_x = self.encdec.initial_encode(x)
        n_batch = images.shape[0]
                
        x_0 = self.encdec(init_x, g_0, conditioning, deterministic)
        x_1 = self.encdec(init_x, g_1, conditioning, deterministic)        
        
        # 1. RECONSTRUCTION LOSS
        if self.ddpm_config.no_recon_loss:
            loss_recon = 0.
        else:            
            # add noise and reconstruct
            eps_0 = jax.random.normal(self.make_rng('sample'), shape = init_x.shape)
            # q(z_t|x) = N(alpha_t*x, sigma_t**2I), alpha_t = sqrt(1 - sigma_t**2)
            # z_0 = jnp.sqrt(1. - var_0) * x_0 + jnp.sqrt(var_0) * eps_0 
            # z_0 = alpha_0 * x_0 + jnp.sqrt(var_0) * eps_0 
            # z_0_rescaled = z_0 /alpha_0    
            # SNR(t) = exp(-g_t), jnp.sqrt(var_0)/alpha_0 = SNR(t)**(-0.5)
            z_0_rescaled = x_0 + jnp.exp(0.5 * g_0) * eps_0
            # The decoder expects values in the interval (-1, 1), so z_0 must be 
            # rescaled accordingly
            loss_recon = - self.encdec.logprob(x, z_0_rescaled, g_0)
        
        # 2. LATENT LOSS / PRIOR LOSS 
        # KL z1 with N(0,1) prior, D_KL(q(z_1|x_1) || p(z_1))
        # Use q(z_1|x_1) and p(z_1) are both normal distributions to get
        alpha_x_sqr = (1. - var_1) * jnp.square(x_1) # (alpha * x)**2 
        loss_klz = 0.5 * jnp.sum(alpha_x_sqr + var_1 - jnp.log(var_1) - 1., 
                                 axis = (1, 2, 3))
        
        # 3. DIFFUSION LOSS
        # sample time steps 
        rng1 = self.make_rng('sample')
        if len(ts) != n_batch:          
            t_base = jax.random.uniform(rng1)
            t = jnp.mod(t_base + jnp.arange(0., 1., step = 1./n_batch), 1.)
            #TODO: check whether we can record loss per t
            t_filters = jnp.zeros((4, n_batch), dtype=bool)
            t_step_size = 0.25
            t_steps = jnp.arange(0, 1, t_step_size)
            for i in jnp.arange(4):
                t_filters = t_filters.at[i].set((t_steps[i] < t) & (t < t_steps[i] + t_step_size))
        else:
            t = ts            
                
            
        # Sample z_t 
        g_t = self.gamma(t)        
        x_t = self.encdec(init_x, g_t, conditioning, deterministic)
            
        var_t = nn.sigmoid(g_t).reshape(-1, 1, 1, 1)
        eps = jax.random.normal(self.make_rng('sample'), shape = x_t.shape)
        # q(z_t|x) = N(alpha_t*x_t, sigma_t**2I), alpha_t = sqrt(1 - sigma_t**2)
        z_t = jnp.sqrt(1. - var_t) * x_t + jnp.sqrt(var_t) * eps
        v_t = jnp.sqrt(1. - var_t) * eps - jnp.sqrt(var_t) * x_t
        # Compute predicted v_t
        v_hat = self.score_model(z_t, g_t, conditioning, deterministic)
        
                
        SNR_t = jnp.exp(-g_t)
        SNR_t_min_mean_max = (jnp.min(SNR_t), jnp.mean(SNR_t), jnp.max(SNR_t))
        
        loss_diff_mse = jnp.sum(jnp.square(v_t - v_hat), axis = [1, 2, 3])        
        eps_m_eps_hat = (jnp.min(loss_diff_mse), jnp.mean(loss_diff_mse), jnp.max(loss_diff_mse))
        
        
        # We always work with continuous time
        # loss for continuous time 
        _, g_t_grad = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))
        
        y_x_t = self.encdec.inner_encoder(init_x, g_t, conditioning, deterministic)
        y_x_t_fun = lambda u: self.encdec.inner_encoder(init_x, u, conditioning, deterministic)
        
        x_g_t_fun = lambda u: self.encdec(init_x, u, conditioning, deterministic)
        
        _, x_t_grad = jax.jvp(x_g_t_fun, (g_t,), (jnp.ones_like(g_t),))            
        non_zero_x_t_grad = (x_t_grad != 0.).sum()
        
        _, y_x_t_grad = jax.jvp(y_x_t_fun, (g_t,), (jnp.ones_like(g_t),))  
        
        
        SNR_t_x_t_grad = y_x_t
        SNR_t_x_t_grad_mmm = (jnp.min(SNR_t_x_t_grad), jnp.mean(SNR_t_x_t_grad), jnp.max(SNR_t_x_t_grad))
        
        sigma_t = jnp.sqrt(var_t)
        alpha_squared = 1. - var_t
        x_t_hat = jnp.sqrt(alpha_squared)*z_t - sigma_t * v_hat
        
        norm_loss = jnp.sum(jnp.square(v_t - v_hat + sigma_t * (x_t_hat - x_t + y_x_t + y_x_t_grad)), axis = (1, 2, 3))        
        
        loss_diff = 0.5 * g_t_grad * jnp.squeeze(alpha_squared) * norm_loss
                
        # Not used in this version
        g_t_grad_min_mean_max = (jnp.min(y_x_t_grad), jnp.mean(y_x_t_grad), jnp.max(y_x_t_grad))
        x_t_grad_min_mean_max = (jnp.min(x_t_grad), jnp.mean(x_t_grad), jnp.max(x_t_grad))
        
        if len(ts) != n_batch: 
            t_diff_losses_arr = jnp.zeros((4, int(n_batch/4)))
            for i in jnp.arange(4):
                filt_indexes = jnp.nonzero(t_filters[i], size = int(n_batch/4))
                # if len(t_diff_losses_arr[i]) == len(filt_indexes):
                t_diff_losses_arr = t_diff_losses_arr.at[i].set(loss_diff[filt_indexes])
                
            
            t_diff_losses = (t_diff_losses_arr[0], 
                             t_diff_losses_arr[1], 
                             t_diff_losses_arr[2], 
                             t_diff_losses_arr[3])
        else:
            t_diff_losses = (0,0,0,0)
        
        return VDMOutput(
            loss_recon = loss_recon,
            loss_klz = loss_klz,
            loss_diff = loss_diff,
            var_0 = var_0,
            var_1 = var_1,
            gamma_grad_min_mean_max = g_t_grad_min_mean_max,
            SNR_t_min_mean_max = SNR_t_min_mean_max,
            SNR_t_times_x_t_grad_norm_min_mean_max = SNR_t_x_t_grad_mmm,
            eps_m_eps_hat_norm_min_mean_max = eps_m_eps_hat,
            t_diff_loss = t_diff_losses,
            non_zero_x_t_grad = non_zero_x_t_grad,
            x_t_grad_min_mean_max = x_t_grad_min_mean_max)
    
    
    def encode_decode_to_logprobs(self, x: NDArray) -> Array:
        gamma_min = self.gamma(0.)
        init_x = self.initial_encode(x)
        
        return self.encdec.decode_to_logprobs(init_x, gamma_min)
    
    
    def initial_encode(self, x: NDArray):
        init_x = self.encdec.initial_encode(x)
                
        return init_x
    
    
    def encode(self, x: NDArray, t: float, conditioning: Array, deterministic: bool) -> Array:
        init_x = self.initial_encode(x)
        
        g_t = self.gamma(t)
        
        return self.encdec(init_x, g_t, conditioning, deterministic)
    
    
    def encode_from_init_gt(self, init_x: NDArray, g_t: float, conditioning: Array, deterministic: bool) -> Array:
        return self.encdec(init_x, g_t, conditioning, deterministic)
    
    
    def inner_encode(self, x: NDArray, t: float, conditioning: Array, deterministic: bool) -> Array:
        init_x = self.initial_encode(x)
        
        g_t = self.gamma(t)
        
        return self.encdec.inner_encoder(init_x, g_t, conditioning, deterministic)
    
    
    def get_x_t_grad(self, init_x: Array, g_t: float, conditioning: Array, deterministic: bool):
        x_g_t_fun = lambda u: self.encdec(init_x, u, conditioning, deterministic)
        
        _, x_t_grad = jax.jvp(x_g_t_fun, (g_t,), (jnp.ones_like(g_t),))  
        
        return x_t_grad
    
    
    def encode_to_rgb(self, x: NDArray, t: float, conditioning: Array, deterministic: bool) -> Array:
        encoded = self.encode(x, t, conditioning, deterministic)
        
        as_rgb = self.encdec.x_t_to_rgb(encoded)
        return as_rgb
    
    
    def logprob(self, x: NDArray, z_0_rescaled: Array, g_0: Array) -> Array:
        return self.encdec.logprob(x, z_0_rescaled, g_0)
    
    
    def get_gamma(self, t: Array):
        return self.gamma(t)
    
    
    def score(self, 
              z_t: Array, g_t: Array, conditioning: Array, 
              deterministic: bool = True):
        return self.score_model(z_t, g_t, conditioning, deterministic)
    
    
    def sample(self, i: int, T: int, 
               z_t: Array, 
               conditioning: Array, 
               rng: PRNGKey,
               dynamic_thresholding: bool = False,
               percentile: float = 95.0,
               percentile_scale: float = 0.0,
               scale_x_hat: bool = False):
        rng_body = jax.random.fold_in(rng, i)
        eps = jax.random.normal(rng_body, z_t.shape)
        
        t = (T - i) / T
        s = (T - i - 1) / T
                
        g_s, g_t = self.gamma(s), self.gamma(t)
        v_hat = self.score_model(
            z_t,
            g_t * jnp.ones((z_t.shape[0],)),
            conditioning,
            deterministic = True)
        # mu_P = \alpha_s/\alpha_t * exp(g_s - g_t) * z_t 
        #   + \alpha_s(1 - exp(g_s - g_t)) * x_hat + \alpha_s * (g_s - g_t) * -\sigma_t**2 * x_hat
        # mu_P = \alpha_s/\alpha_t * (exp(g_s - g_t) - 1 + 1) * z_t 
        #   - \alpha_s(exp(g_s - g_t) - 1) * x_hat - \alpha_s * (g_s - g_t) * \sigma_t**2 * x_hat
        # where
        # x_hat = \alpha_t*z_t - \sigma_t * v_hat
        var_t = nn.sigmoid(g_t)
        var_s = nn.sigmoid(g_s)
        sigma_t = jnp.sqrt(var_t)
        sigma_s = jnp.sqrt(var_s)
        alpha_t = jnp.sqrt(1.- var_t)
        alpha_s = jnp.sqrt(1.- var_s)
        # SNR(t)/SNR(s) = exp(g_s - g_t)
        snr_t_s_m1 = jnp.expm1(g_s - g_t)
        
        x_hat = alpha_t*z_t - sigma_t * v_hat
        
        if dynamic_thresholding:
            p_val = jnp.percentile(jnp.abs(x_hat), percentile)
            if p_val > 1:
                x_hat = jnp.clip(x_hat, a_min = -p_val, a_max = p_val)/p_val
        elif scale_x_hat:
            x_hat_max = jnp.max(jnp.abs(x_hat), axis = None)
            if x_hat_max > 1:
                x_hat = x_hat / x_hat_max
        
        mu_P = alpha_s/alpha_t * (snr_t_s_m1 + 1.) * z_t - alpha_s * snr_t_s_m1 * x_hat - alpha_s * (g_s - g_t) * sigma_t**2 * x_hat
        sigma_Q = sigma_s * jnp.sqrt((-1.) * snr_t_s_m1) 
        
        z_s = mu_P + sigma_Q * eps            
        
        
        if percentile_scale > 0:
            # percentile_scale is percentile we want to be 1
            p_val = jnp.percentile(jnp.abs(z_s), percentile)
            if p_val > 1:
                z_s = z_s/p_val            
        
        
        return z_s
    
    
    def generate_x(self, z_0):        
        gamma_0 = self.gamma(0.)
        if self.ddpm_config.sample_softmax:
            out_rng = self.make_rng('sample')
        else:
            out_rng = None
        
        return self.encdec.generate_x(z_0, gamma_0, out_rng)
    
    
    def z_t_for_visualisation(self, z_t: Array):
        # TODO: Fix this since z_t might not have values close to (-1, 1) 
        return self.generate_x(z_t)
    






























