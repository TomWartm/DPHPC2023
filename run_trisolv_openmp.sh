#!/bin/bash

module load gcc openblas openmpi
make evaluate_trisolv_openmp

for thread in 1 2 4 8 16 32 48
do
    export OMP_NUM_THREADS=$thread
    sbatch --ntasks=1 --cpus-per-task=$thread --wrap="./evaluate_trisolv_openmp"
done
