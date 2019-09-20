#!/bin/bash
#SBATCH --job-name="rdy1kmsra"
#SBATCH --workdir=.
#SBATCH --output=rdy1kmsra.out
#SBATCH --error=rdy1kmsra.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python restore_model_dynamic.py solver_dynamic_ema_1000_batches_msra.prototxt 100 dyn_ema_1000_batches_res_50_iter_132000.solverstate
