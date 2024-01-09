#!/bin/bash

module load gcc openblas openmpi
make evaluate_trisolv_openmp
make evaluate_gemver_openmp
make evaluate_gemver_mpi
make evaluate_trisolv_mpi

# OpenMP -> Max Threads
export OMP_NUM_THREADS=32
sbatch --ntasks=1 --time=120 --cpus-per-task=32 --wrap="./evaluate_trisolv_openmp"
sbatch --ntasks=1 --time=120 --cpus-per-task=32 --wrap="./evaluate_gemver_openmp"

# MPI -> Max Tasks, 1 core
export OMP_NUM_THREADS=1
sbatch --ntasks=32 --time=120 --cpus-per-task=1 --wrap="mpirun -np 32 ./evaluate_trisolv_mpi"
sbatch --ntasks=32 --time=120 --cpus-per-task=1 --wrap="mpirun -np 32 ./evaluate_gemver_mpi"

# MPI, OpenMP Hybrid,
export OMP_NUM_THREADS=8
sbatch --ntasks=4 --time=120 --cpus-per-task=8 --wrap="mpirun -np 4 ./evaluate_trisolv_mpi"
sbatch --ntasks=4 --time=120 --cpus-per-task=8 --wrap="mpirun -np 4 ./evaluate_gemver_mpi"
