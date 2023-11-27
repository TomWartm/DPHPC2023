#ifndef GEMVER_MPI_BLOCKING_H
#define GEMVER_MPI_BLOCKING_H

// Function to perform the GEMVER operation

void gemver_mpi_4(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z, double *A_result, double *x_result, double *w_result);

#endif 