#!/bin/bash -login

#SBATCH --job-name=train_pose
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-24:00:00

module load CUDA
module load apps/ffmpeg/4.3

echo "Training pose..."
python train_DI_pose.py
