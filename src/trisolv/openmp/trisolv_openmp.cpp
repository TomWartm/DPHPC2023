#include "omp.h"
#include <iostream>
#include <cstring>
#include <algorithm>

#define PAD 8

void trisolv_openmp(int n, double* L, double* x, double* b) {
    omp_set_num_threads(NUM_THREADS);
    memcpy(x, b, (sizeof (double)) * n);
    int limit = std::min(NUM_THREADS * 16, n);
    double test[limit][PAD];
    for(int i = 0; i < limit; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum -= L[i * n + j] * test[j][0];
        }
        test[i][0] = (b[i] + sum) / L[i * n + i];
    }

    #pragma omp parallel for
    for(int i = 0; i < limit; i++) {
        x[i] = test[i][0];
    }

    for (int i = NUM_THREADS * 16; i < n; i++) {
        double sums[NUM_THREADS][PAD];
        int block_size = i / NUM_THREADS;
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            int start = id * block_size;
            int end;
            if (id == NUM_THREADS - 1) {
                end = i;
            } else {
                end = start + block_size;
            }
            sums[id][0] = 0.0;
            for (int j = start; j < end; j++) {
                sums[id][0] += -L[i * n + j] * x[j];
            }
        }
        for (int j = 0; j < NUM_THREADS; j++) {
            x[i] += sums[j][0];
        }
        x[i] = x[i] / L[i * n + i];
    }
}

void trisolv_openmp_2(int n, double* L, double* x, double* b) {
    omp_set_num_threads(NUM_THREADS);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < i; j++) {
            sum -= L[i * n + j] * x[j];
        }
        x[i] = (b[i] + sum) / L[i * n + i];
    }
}

void trisolv_openmp_lowspace(int n, double* L, double* x, double* b) {
    omp_set_num_threads(NUM_THREADS);
    memcpy(x, b, (sizeof (double)) * n);
    for (int i = 0; i < n; i++) {
        int index = ((i + 1) * i) >> 1;
        double sums[NUM_THREADS][PAD];
        int block_size = i / NUM_THREADS;
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            int start = id * block_size;
            int end;
            if (id == NUM_THREADS - 1) {
                end = i;
            } else {
                end = start + block_size;
            }
            sums[id][0] = 0.0;
            for (int j = start; j < end; j++) {
                sums[id][0] += -L[index + j] * x[j];
            }
        }
        for (int j = 0; j < NUM_THREADS; j++) {
            x[i] += sums[j][0];
        }
        x[i] = x[i] / L[index + i];
    }
}

void trisolv_openmp_3(int n, double* L, double* x, double* b) {
    omp_set_num_threads(NUM_THREADS);

    for (int i = 0; i < n; i++) {
        x[i] = b[i] / L[i * n + i];
        double tmp = x[i];
        #pragma omp parallel for
        for(int j = i + 1; j < n; j++) {
            x[j] -= L[i * n + j] * tmp;
        }
    }
}