#!/bin/bash

module load gcc openblas openmpi
make evaluate_trisolv_mpi

for cores in 1 2 4 8 16 32 48
do
  for ((tasks = 2; tasks <= cores; tasks += 2)); do
      cpus=$((thread - tasks))
      export OMP_NUM_THREADS=$cpus
      sbatch --ntasks=$tasks --cpus-per-task=$cpus --wrap="mpirun -np $tasks ./evaluate_trisolv_mpi"
  done
done
