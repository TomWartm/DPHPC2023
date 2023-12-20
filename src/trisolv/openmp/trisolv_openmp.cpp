#include "omp.h"
#include <iostream>

#define PAD 8

void trisolv_openmp(int n, double* L, double* x, double* b) {
    int num_threads;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }

    for (int i = 0; i < n; i++) {
        double sums[num_threads][PAD];

        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            //Ceiling division
            int block_size = (i + num_threads - 1) / num_threads;
            int start = id * block_size;
            //I changed this
            int end = start + block_size;
            if (end > i) {
                end = i;
            }
            sums[id][0] = 0.0;
            for (int j = start; j < end; j++) {
                sums[id][0] += -L[i * n + j] * x[j];
            }
        }
        x[i] = b[i];
        for (int j = 0; j < num_threads; j++) {
            x[i] += sums[j][0];
        }
        x[i] /= L[i * n + i];
    }
}

void trisolv_openmp_2(int n, double* L, double* x, double* b) {

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < i; j++) {
            sum -= L[i * n + j] * x[j];
        }
        x[i] = (b[i] + sum) / L[i * n + i];
    }
}

void trisolv_openmp_3(int n, double* L, double* x, double* b) {

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        if(i > 4000) {
            #pragma omp parallel for reduction(+:sum)
            for (int j = 0; j < i; j++) {
                sum -= L[i * n + j] * x[j];
            }
            x[i] = (b[i] + sum) / L[i * n + i];
        } else {
            x[i] = b[i];
            for (int j = 0; j < i; j++) {
                x[i] = -L[i * n + j] * x[j];
            }
            x[i] = x[i] / L[i * n + i];
        }
    }
}