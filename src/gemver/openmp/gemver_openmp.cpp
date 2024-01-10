#include "omp.h"
#include <iostream>

#define PAD 8

//baseline implementation
void gemver_openmp_v0(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = A[i * n + j] + u1[i] * v1[j] + u2[i] * v2[j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            x[i] = x[i] + beta * A[j * n + i] * y[j];

    for (int i = 0; i < n; i++)
        x[i] = x[i] + z[i];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            w[i] = w[i] + alpha * A[i * n + j] * x[j];
}

//OpenMP simple version
void gemver_openmp_v1(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        double sum = 0.0;
        for (int j = 0; j < n; j++){
            sum += beta * A[j * n + i] * y[j];
        }
        x[i] += sum + z[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += alpha * A[i * n + j] * x[j];
        }
        w[i] += sum;
    }
}

//OpenMP with blocking except first
void gemver_openmp_v2(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    const int block_size = 64/sizeof(double);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
        }
    }


    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sum = 0.0;

        for (int jblock = 0; jblock < n; jblock += block_size) {
            int jmax = jblock + block_size < n ? jblock + block_size : n;
            for (int j = jblock; j < jmax; j++) {
                sum += beta * A[j * n + i] * y[j];
            }
        }
        x[i] += sum + z[i];
    }

    #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int jblock = 0; jblock < n; jblock += block_size) {
                int jmax = jblock + block_size < n ? jblock + block_size : n;
                for (int j = jblock; j < jmax; j++) {
                    w[i] += alpha * A[i * n + j] * x[j];
                }
            }
        }
    }


//OpenMP with blocking
void gemver_openmp_v3(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    const int block_size = 64/sizeof(double);

    #pragma omp parallel for
        for (int iblock = 0; iblock < n; iblock += block_size) {
            int imax = std::min(iblock + block_size, n);
            for (int jblock = 0; jblock < n; jblock += block_size) {
                int jmax = std::min(jblock + block_size, n);
                for (int i = iblock; i < imax; i++) {
                    for (int j = jblock; j < jmax; j++) {
                        A[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
                    }
                }
            }
        }


    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sum = 0.0;

        for (int jblock = 0; jblock < n; jblock += block_size) {
            int jmax = jblock + block_size < n ? jblock + block_size : n;
            for (int j = jblock; j < jmax; j++) {
                sum += beta * A[j * n + i] * y[j];
            }
        }
        x[i] += sum + z[i];
    }
 
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int jblock = 0; jblock < n; jblock += block_size) {
            int jmax = jblock + block_size < n ? jblock + block_size : n;
            for (int j = jblock; j < jmax; j++) {
                w[i] += alpha * A[i * n + j] * x[j];
            }
        }
    }    
}


//OpenMP with reduction of the sum
void gemver_openmp_v4(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    double sum = 0.0;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += beta * A[j * n + i] * y[j];
        }
        x[i] += sum +  z[i];
    }

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += alpha * A[i * n + j] * x[j];
        }
        w[i] += sum;
    }
}
