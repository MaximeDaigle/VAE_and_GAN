#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
echo $SLURM_TMPDIR
source $SLURM_TMPDIR/env/bin/activate

#pip install --no-index numpy==1.15.0
#pip install torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl

pip install --no-index torch
pip install --no-index torchvision
pip install --no-index scipy

echo ""
echo "Calling python train script."
stdbuf -oL python -u train.py
