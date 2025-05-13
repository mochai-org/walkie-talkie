#!/bin/sh
#SBATCH -p hgx
#SBATCH -w hgx2
#SBATCH -c52
#SBATCH -A mgr_wt
#SBATCH --gres=gpu:1

cd /home/inf151841/nanochi/ai2/ai
eval "$(/home/inf151841/anaconda3/bin/conda shell.bash hook)"

python train2.py
