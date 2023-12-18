#include "omp.h"
#include <iostream>

#define PAD 8



//Used the baseline implementation as a place holder
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

//OpenMP without padding
void gemver_openmp_v1(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    omp_set_num_threads(NUM_THREADS);
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
        x[i] += sum;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        x[i] += z[i];
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

//OpenMP with padding
void gemver_openmp_v2(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    omp_set_num_threads(NUM_THREADS);
    double sums[NUM_THREADS][PAD];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
        }
    }
    for (int i = 0; i < n; i++){
        int block_size = n / NUM_THREADS;
        omp_set_num_threads(NUM_THREADS);
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            int start = id * block_size;
            int end = (id == NUM_THREADS - 1) ? n : start + block_size;
            sums[id][0] = 0.0;
            for (int j = start; j < end; j++){
                sums[id][0] += beta * A[j * n + i] * y[j];
            }
        }
        for (int j = 0; j < NUM_THREADS; j++){
            x[i] += sums[j][0];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        x[i] += z[i];
    }

    for (int i = 0; i < n; i++){
        double sums[NUM_THREADS][PAD];
        int block_size = n / NUM_THREADS;
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            int start = id * block_size;
            int end = (id == NUM_THREADS - 1) ? n : start + block_size;
            sums[id][0] = 0.0;
            for (int j = start; j < end; j++) {
                sums[id][0] += alpha * A[i * n + j] * x[j];
            }
        }
        for (int j = 0; j < NUM_THREADS; j++){
            w[i] += sums[j][0];
        }
    }
}