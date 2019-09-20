#!/bin/bash
#SBATCH --job-name="dyem1kmsra"
#SBATCH --workdir=.
#SBATCH --output=dyem1kmsra.out
#SBATCH --error=dyem1kmsra.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
export OMP_NUM_THREADS=48

srun pin -inline 1 -t pintool_path/dynamic.so -- python trainResNetDynamic.py solver_dynamic_ema_1000_batches_msra.prototxt 100
