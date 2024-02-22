# Project in Design of Paralell and High Performance Computing (DPHPC) 2023
## Polybench Parallelization of Gemver and Trisolv

(Abstract) 

In this work, we address the efficiency of solving large-scale computing problems by parallelizing and optimizing two benchmarks, Gemver and Trisolv, from the Polybench collection. We employ parallelization using Message Passing Interface (MPI) and OpenMP, aiming to reduce cache misses and floating-point operations. Performance evaluations were conducted on the ETH cluster Euler, with a varying number of input sizes, processors, and threads. Our OpenMP implementation for Gemver performs similarly to OpenBLAS when utilizing higher thread numbers, whereas our MPI implementation for Trisolv surpasses the performance of the OpenBLAS routine by nearly 1.5 times and achieves a speedup of 7x over the baseline

(see `report.pdf` for full text)

# Generate plots

Create the plot data:
```
make trisol_openmp
make trisolv_mpi

make gemver_openmp
make gemver_mpi
```

Generate the plot:
```
python plot/plot.py
```

# Run Tests

- Trisolv:  `make test_trisolv_openmp`
- Gemver:  `make test_gemver_openmp`


- Trisolv MPI:  `make test_trisolv_mpi`
- Gemver MPI:  `make test_gemver_mpi`

# How to run jobs on Euler

To connect use ``ssh username@euler.ethz.ch``. To send files over use ``scp file username@euler.ethz.ch:/cluster/home/username/`` (or ``cluster/scratch/username``).

Jobs have to be submitted via sbatch. Either ``sbatch --wrap="./a.out"`` or ``sbatch run.sh`` to submit a script. This includes compiling.

## ompenMPI

First you need to load the right modules. Use ``env2lmod`` to make sure that you are using the new software stack. Load the compiler with ``module load .gcc/x.y.z`` and then MPI with ``module load openmpi/x.y.z``.

A program is run with ``sbatch -C ib --ntasks=4 --wrap="mpirun ./a.out"`` (no need to use ``-np 4``).

The sbatch option ``-C ib`` is supposed to ensure faster connection between the nodes.

## openMPI and openMP

A bash script to run a hybrid job on 2 nodes (Each node has 2 sockets), with 8 mpi processes and 6 openMP threads per process would look like this:
```
#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-socket=2
#SBATCH --cores-per-socket=12
#SBATCH --nodes=2
export OMP_NUM_THREADS=6
export OMP_PLACES=cores
srun --cpu-per-task=6 a.out
```
This works for OpenMPI >= 4.1.4


## Evaluate gemver


How I evaluate gemver on euler sth. all experiments are run on the same set of machines:
0) copy entire code
1) import modules:
```
module load gcc
module load openmpi
```
2) compile: 
```
export  OMP_NUM_THREADS=2
make gemver_mpi
make gemver_openmp
```  
3) run:
```
sbatch -C ib --ntasks=8 --mem-per-cpu=16G --cpus-per-task=2 --wrap="mpirun ./evaluate_gemver_mpi; ./evaluate_gemver_openmp"
```  


