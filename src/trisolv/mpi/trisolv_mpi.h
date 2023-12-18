#ifndef TRISOLV_MPI_H
#define TRISOLV_MPI_H

void trisolv_mpi_v0(int n, double* L, double* x, double* b);

void trisolv_blas(int n, double* L, double* x, double* b);

void trisolv_mpi_isend(int n, double* L, double* x, double* b);

void trisolv_mpi_onesided(int n, double* L, double* x, double* b);

void trisolv_mpi_gao(int n, double* A, double* x, double* b);

void trisolv_mpi_onesided_openmp(int n, double* L, double* x, double* b);

void trisolv_mpi_isend_openmp(int n, double* L, double* x, double* b);

#endif //TRISOLV_MPI_H
