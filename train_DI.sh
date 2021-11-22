#!/bin/bash -login

#SBATCH --job-name=PS1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-24:00:00

#module load CUDA
#module load apps/ffmpeg/4.3
module load apps/torch/28.01.2019

export PATH="/mnt/storage/home/km14740/miniconda3/bin:$PATH"

echo "Training pose..."
source /mnt/storage/home/km14740/miniconda3/bin/activate ratslam

python train_DI_pose.py --name "POSTSKIP" --trainkey "top" --simpen 0 --epochs 1000