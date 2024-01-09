#include "omp.h"
#include <iostream>

#define PAD 8

void trisolv_openmp(int n, double* L, double* x, double* b) {
    int num_threads = omp_get_max_threads();
    for (int i = 0; i < n; i++) {
        double sums[num_threads][PAD];
        int block_size = i / num_threads;
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            int start = id * block_size;
            int end;
            if (id == num_threads - 1) {
                end = i;
            } else {
                end = start + block_size;
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
        x[i] = x[i] / L[i * n + i];
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