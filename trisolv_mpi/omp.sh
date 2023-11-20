#!/bin/bash
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-socket=16
#SBATCH --cores-per-socket=256
#SBATCH --nodes=1
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
srun --cpus-per-task=16 ./trisolv_colmaj_partialInit_omp.cpp

