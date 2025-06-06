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
python ps_color_synth_extension.py\
  --input_dir "../../Datasets/select_color_textures_unsplash/" \
  --output_dir "output/" \
  --max_iter 5000 \
  --store_progress 100 \
  --scale_ch_covar 1.0 \
  --scale_ch_mag 1.0 \
  --scale_ch_real 1.0 \
  --stop_criterion 1e-6