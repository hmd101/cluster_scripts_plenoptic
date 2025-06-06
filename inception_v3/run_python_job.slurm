#!/bin/bash

# Request 1 nodes that will not be allocated to others for the duration of the run.
#SBATCH -p gpu -N1 -G1 --constraint=a100 --cpus-per-task=4 --mem-per-cpu=8G --exclusive

# Find out what nodes we were assigned.
srun hostname

# Load modules
module load python/3.10
module load cuda
module load cudnn
module load nccl
module load python

# Activate python virtual environment
source ~/.virtualenvs/plenoptic/bin/activate

# Run python script
python inception_metamers.py 