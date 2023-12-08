#ifndef GEMVER_MPI_H
#define GEMVER_MPI_H

// Function to perform the GEMVER operation
void gemver_mpi_1(int n, double *A_result, double *x_result, double *w_result);
void gemver_mpi_2(int n, double *A_result, double *x_result, double *w_result);
void gemver_mpi_3(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z, double *A_result, double *x_result, double *w_result);
void gemver_mpi_v1(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z, double *A_result, double *x_result, double *w_result);
void gemver_mpi_2_new(int n, double *A_result, double *x_result, double *w_result);
#endif