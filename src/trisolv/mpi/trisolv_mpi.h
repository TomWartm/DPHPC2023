#ifndef GEMVER_MPI_H
#define GEMVER_MPI_H


void kernel_trisolv_mpi(int n, double* L, double* x, double* b);

void kernel_trisolv_mpi_onesided(int n, double* L, double* x, double* b);

#endif 