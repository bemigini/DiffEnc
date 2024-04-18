# Denoising Diffusion Model with Time Dependent Encoder
Code for the article: [DiffEnc: Variational Diffusion with a Learned Encoder](https://arxiv.org/abs/2310.19789) by Beatrix M. G. Nielsen, Anders Christensen, Andrea Dittadi and Ole Winther. Article accepted for publication at ICLR 2024.

Checkpoints and samples can be found at: COMING

## Description
We explore the consequences of adding a time-dependent encoder to a diffusion model. In the case of a trainable encoder we can get an improved likelihood on CIFAR-10. We note that the way we choose to parameterize the trainable encoder enables the model to achieve a better latent loss without harming the diffusion loss. We do experiments on MNIST, CIFAR-10 and ImageNet32. 

Code for generating the figures in the article is in article_figures.py. Note that this requires the relevant checkpoints and sample files.  
 

## Installation guide

The outline of the installation is the following:

**1. Create and activate conda environment**

**2. Conda install relevant conda packages**

**3. Install jax and jaxlib corresponding to your CUDA version**

**4. Pip install relevant pip packages**

In 2. and 3. there might be differences depending on your machine and preferences.

**1. Create and activate conda environment**

Use the commands:
```
conda create -n jax-pip python=3.9
conda activate jax-pip
```

**2. Conda install relevant conda packages** 

If you want to install CUDA in your conda environment, this is where you need the command:
```
conda install -c "nvidia/label/cuda-xx.x.x" cuda-nvcc 
```
Where xx.x.x should be replaced with the cuda version you want. For example:
```
conda install -c "nvidia/label/cuda-12.1.1" cuda-nvcc 
```
If you are not installing CUDA in your conda environment, make sure it is activated, e.g. using module load. 

Use the command:
```
conda install tensorflow docopt tqdm tensorflow-datasets scikit-image=0.20.0 matplotlib=3.7.1 
```

**3. Install jax and jaxlib corresponding to your CUDA version** 

See https://github.com/google/jax#installation for choosing the correct one. 
Below an example for installing wheel (only for linux) compatible with CUDA 12 and cuDNN 8.8 or newer:
```
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**4. Pip install relevant pip packages** 

Use the command:
```
pip install flax==0.6.10 tensorflow-io-gcs-filesystem libclang tensorstore jaxtyping==0.2.20
```


## Datasets
MNIST and CIFAR-10 will be automatically loaded from tensorflow_datasets. 
ImageNet32 must be downloaded from the [official ImageNet webpage](https://image-net.org/download-images.php). Note that this requires a user. 


## Usage
Config files used to train the models from the article can be found in the config folder. 
The baseline model, a VDM with v parametrization, is called VDM2_V.
The model with a non-trainable time dependent encoder is called S4V_NT.
The model with a non-trainable time dependent encoder is called S3VA. 


To see all options, use:
```
python run.py -h
```
To train models use:
```
python run.py train --output-folder=<file> --config-folder=<file> --c-model=<file> --c-opt=<file> --c-train=<file> --train-steps=<int> [options]
```
Where output-folder is where checkpoints and metrics will be saved, config-folder is the location of the config files, c-model is the file name of the model config, c-opt is the file name of the optimizer config, c-train is the file name for the training config and train-steps is the number of training steps. Options such as train-seed can be added.    
For example:
```
python run.py train --output-folder=/scratch/diffenc --config-folder=config --c-model=config_model_VDM2_V_fixed_-13_3_5_0_unet_8_simple.json --c-opt=config_opt_adamw_0002_no_clip.json --c-train=config_train_cifar10_unconditional_1_128.json --train-steps=800000 --train-seed=1
```
Will train a VDM model with v parametrization for 800K steps on CIFAR-10 using the seed 1. Training can be resumed from a saved checkpoint by adding the date prefix, date-str, of the checkpoint. For example:  
```
python run.py train --output-folder=/scratch/diffenc --config-folder=config --c-model=config_model_S3VA_fixed_-13_3_5_0_unet_8_unet_2_m1_to_1.json --c-opt=config_opt_adamw_0002_no_clip.json --c-train=config_train_cifar10_unconditional_1_128.json --date-str=2024-02-07 --train-steps=800000 --train-seed=1
```
Will load a checkpoint of a model with a trainable encoder which began training on 2024-02-07 and will continue training this model for another 800K steps. 

When training on ImageNet32, one must also supply the folder where the ImageNet dataset can be found. For example: 
```
python run.py train --output-folder=/scratch/diffenc --config-folder=config --c-model=config_model_S3VA_fixed_-13_3_5_0_unet_32_unet_mult_res_8_m1_to_1_ImageNet.json --c-opt=config_opt_adamw_0002_no_clip.json --c-train=config_train_imagenet32_unconditional_1_256.json --train-steps=800000 --train-seed=1 --imagenet-folder=/scratch/ImageNet
```
Example of a bash script to train a model on ImageNet is in bash_run_example.s 

For evaluating models use:
```
python run.py evaluate --output-folder=<file> --config-folder=<file> --c-model=<file> --c-opt=<file> --c-train=<file> --date-str=<string> --train-steps=<int> [options]
```
For sampling use: 
```
python run.py sample --output-folder=<file> --config-folder=<file> --c-model=<file> --c-opt=<file> --c-train=<file> --date-str=<string> --train-steps=<int> [options]
```

## Authors and acknowledgment
This work was supported by the Danish Pioneer Centre for AI, DNRF grant number P1, and by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254). OW’s work was funded in part by the Novo Nordisk Foundation through the Center for Basic Machine Learning Research in Life Science (NNF20OC0062606). AC thanks the ELLIS PhD program for support. 

## License and Copyright
Copyright © 2023 Technical University of Denmark

See LICENSE.



