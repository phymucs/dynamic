#!/bin/bash
#SBATCH --job-name="rdy1kalex"
#SBATCH --workdir=.
#SBATCH --output=rdy1kalex.out
#SBATCH --error=rdy1kalex.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python restore_model_dynamic_alexnet.py solver_alexnet_dynamic_ema_1000_batches.prototxt 100 alexnet_dynamic_ema_1000_iter_44000.solverstate 62
