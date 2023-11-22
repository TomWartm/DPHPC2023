#ifndef TRISOLV_MPI_H
#define TRISOLV_MPI_H

void trisolv_mpi_v0(int n, double* L, double* x, double* b);

void kernel_trisolv_mpi(int n, double* L, double* x, double* b);

void kernel_trisolv_mpi_onesided(int n, double* L, double* x, double* b);


#endif //TRISOLV_MPI_H