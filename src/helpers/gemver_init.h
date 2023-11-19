#ifndef GEMVER_INIT_H
#define GEMVER_INIT_H


void init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);

void sparse_init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);

void rand_init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);




// You can also declare any other necessary constants, data structures, or functions here

#endif // GEMVER_INIT_H