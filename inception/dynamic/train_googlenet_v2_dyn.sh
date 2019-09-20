#!/bin/bash
#SBATCH --job-name="dyngoo"
#SBATCH --workdir=.
#SBATCH --output=dyngoo.out
#SBATCH --error=dyngoo.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python train_googlenet_v2_dyn.py solver_googlenet_v2_dyn.prototxt 100
