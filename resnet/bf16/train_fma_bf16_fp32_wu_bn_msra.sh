#!/bin/bash
#SBATCH --job-name="fma16msra"
#SBATCH --workdir=.
#SBATCH --output=fma16msra.out
#SBATCH --error=fma16msra.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python train_fma_bf16_fp32_wu_bn_msra.py solver_bf16_fma_fp32_wu_bn_msra.prototxt 3
