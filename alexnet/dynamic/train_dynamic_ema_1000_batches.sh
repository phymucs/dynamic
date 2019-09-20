#!/bin/bash
#SBATCH --job-name="dyem1kalex"
#SBATCH --workdir=.
#SBATCH --output=dyem1kalex.out
#SBATCH --error=dyem1kalex.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python trainAlexNetDynamic.py solver_alexnet_dynamic_ema_1000_batches.prototxt 100
