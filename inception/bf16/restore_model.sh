#!/bin/bash
#SBATCH --job-name="rgoobf16"
#SBATCH --workdir=.
#SBATCH --output=rgoobf16.out
#SBATCH --error=rgoobf16.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python restore_model.py solver_googlenet_v2_bf16_test2.prototxt bf16_googlenet_v2_iter_48000.solverstate
