#!/bin/bash
#SBATCH --job-name="rbf1kalex"
#SBATCH --workdir=.
#SBATCH --output=rbf1kalex.out
#SBATCH --error=rbf1kalex.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python ../restore_model_alexnet.py solver_alexnet_bf16_fma_fp32_wu_bn.prototxt alexnet_bf16_fma_fp32_wu_bn_iter_22000.solverstate 34 2200
