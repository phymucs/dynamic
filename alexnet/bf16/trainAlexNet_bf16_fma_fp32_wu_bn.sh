#!/bin/bash
#SBATCH --job-name="alebf16"
#SBATCH --workdir=.
#SBATCH --output=alebf16.out
#SBATCH --error=alebf16.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python trainAlexNet_bf16_fma_fp32_wu_bn.py solver_alexnet_bf16_fma_fp32_wu_bn.prototxt 10
