#!/bin/bash
#SBATCH --job-name="goobf16"
#SBATCH --workdir=.
#SBATCH --output=goobf16.out
#SBATCH --error=goobf16.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python train_bf16_googlenet_v2.py solver_googlenet_v2_bf16_test2.prototxt 4
