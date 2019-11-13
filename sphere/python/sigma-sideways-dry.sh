#!/bin/bash

#SBATCH -o sigma-sideways-dry.%j.%N.out
#SBATCH -p longq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -J sigma-sideways-dry
#SBATCH --time=48:00:00

module load gcc/4.8.5
module load cuda75

cd ~/code/sphere/python/
python sigma-sideways-starter.py 0 0 1.0 2.0e-16 10000.0 2.080e-7 1.0

