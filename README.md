# Generate plots
1. Compile the project by running `make` in the root directory.
2. Execute the Binary `dphpc` in the root directory.
3. Run `python plot/plot.py` from the root directory.


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
