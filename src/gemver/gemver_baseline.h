#ifndef GEMVER_BASELINE_H
#define GEMVER_BASELINE_H

// Function to perform the GEMVER operation
void kernel_gemver(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);
void kernel_gemver(int n, int m, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);
void gemver_baseline_blocked_1(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);
void gemver_baseline_blocked_2(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);
void gemver_baseline_blocked_2(int n, int m, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z);
void part_1(int n, int m, double *A, double *u1, double *u2, double *v1, double *v2);
void part_2_2(int n, int m, double beta, double *A_transposed, double *x, double *y, double *z);
void part_3(int n,int m, double alpha, double *A, double *x,  double *w );
// You can also declare any other necessary constants, data structures, or functions here

#endif // GEMVER_BASELINE_H