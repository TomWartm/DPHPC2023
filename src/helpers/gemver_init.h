#ifndef GEMVER_INIT_H
#define GEMVER_INIT_H


void init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);
void init_gemver(int n, int m, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);

void sparse_init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);

void rand_init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);
void init_gemver_vy(int n, double *alpha, double *beta, double* v1, double* v2, double* y);
void init_gemver_Au(int n, int m, int n_min, int n_max, double* A, double* u1, double* u2);
void init_gemver_w(int n, double* w);
void init_gemver_xz(int n, int n_min, int n_max, double* x, double* z );


// You can also declare any other necessary constants, data structures, or functions here

#endif // GEMVER_INIT_H