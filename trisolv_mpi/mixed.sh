#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-socket=2
#SBATCH --cores-per-socket=12
#SBATCH --nodes=2
export OMP_NUM_THREADS=6
export OMP_PLACES=cores
srun --cpu-per-task=6 trisolv

