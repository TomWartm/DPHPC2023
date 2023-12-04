#ifndef DPHPC2023_GEMVER_OPENMP_H
#define DPHPC2023_GEMVER_OPENMP_H

//v0 represents the baseline, impelemented as a placeholder for testing purposes
void gemver_openmp_v0(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);
//without padding
void gemver_openmp_v1(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);
//with padding
void gemver_openmp_v2(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);

#endif //DPHPC2023_GEMVER_OPENMP_H
