#!/bin/bash
#SBATCH --job-name="rdyngoo"
#SBATCH --workdir=.
#SBATCH --output=rdyngoo.out
#SBATCH --error=rdyngoo.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python restore_model_dynamic.py solver_googlenet_v2_dyn.prototxt 100 dyn_googlenet_v2_iter_48000.solverstate
