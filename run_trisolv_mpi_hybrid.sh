#!/bin/bash

module load gcc openblas openmpi
make evaluate_trisolv_mpi

for cores in 1 2 4 8 16 32 48
do
  if [ "$cores" -eq 1 ]; then
      export OMP_NUM_THREADS=1
      sbatch --ntasks=1 --cpus-per-task=1 --mem-per-cpu=16G --wrap="mpirun -np 1 ./evaluate_gemver_mpi"
  else
    for ((tasks = 2; tasks <= cores; tasks += 2)); do
        cpus=$((cores / tasks))
        export OMP_NUM_THREADS=$cpus

        mem_per_cpu=$((128 / cores))
        mem_per_cpu=$(( mem_per_cpu > 16 ? 16 : mem_per_cpu ))

        sbatch --ntasks=$tasks --cpus-per-task=$cpus --mem-per-cpu=${mem_per_cpu}G --wrap="mpirun -np $tasks ./evaluate_trisolv_mpi"
    done
  fi
done
