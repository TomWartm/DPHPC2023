#!/bin/bash

module load gcc openblas openmpi
make evaluate_trisolv_mpi

for thread in 1 2 4 8 16 32 48
do
    export OMP_NUM_THREADS=1
    sbatch --ntasks=$thread --cpus-per-task=1 --wrap="mpirun -np $thread ./evaluate_trisolv_mpi"
done
