#!/bin/bash
#SBATCH --job-name="rbfres"
#SBATCH --workdir=.
#SBATCH --output=rbfres.out
#SBATCH --error=rbfres.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python restore_model.py solver_bf16_fma_fp32_wu_bn_msra.prototxt fma_bf16_fp32_wu_bn_res_50_iter_108000.solverstate
