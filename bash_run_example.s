#!/usr/bin/bash
#SBATCH --job-name=VDMVimgnet_-13_3_5_0_13
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 48:00:00



## INFO
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

conda init bash
source ~/.bashrc

cd DiffEnc/
conda activate jax-pip

echo -e "Working dir: $(pwd)\n"

python run.py train --output-folder=/scratch/diffenc --config-folder=config --c-model=config_model_VDM2_V_fixed_-13_3_5_0_unet_32_simple_ImageNet.json --c-opt=config_opt_adamw_0002_no_clip.json --c-train=config_train_imagenet32_unconditional_1_256.json --train-steps=300000 --train-seed=1 --imagenet-folder=/scratch/ImageNet


echo "Done: $(date +%F-%R:%S)"