#include "omp.h"
#include <iostream>

#define PAD 8
#define NUM_THREADS 8

void trisolv_openmp(int n, double* L, double* x, double* b)
{
  for (int i = 0; i < n; i++)
  {
      x[i] = b[i];
      if(i > 10000) {
          int sums[NUM_THREADS][PAD];
          int block_size = i / NUM_THREADS;
          omp_set_num_threads(NUM_THREADS);
          #pragma omp parallel
          {
              int id = omp_get_thread_num();
              int max;
              if (id == NUM_THREADS - 1) {
                  max = i;
              } else {
                  max = block_size * (id + 1);
              }

              //std::cout << printf("Iteration %d: Thread %d: - min: %d max: %d\n", i, id, start, max);
              for (int j = id * block_size; j < max; j++) {
                  sums[id][0] += -L[i * n + j] * x[j];
              }
          }
          for(int j = 0; j < NUM_THREADS; j++) {
              x[i] = x[i] + sums[j][0];
          }
      } else {
          for (int j = 0; j < i; j++) {
              double tmp = -L[i * n + j] * x[j];
              x[i] = x[i] + tmp;
          }
      }

      x[i] = x[i] / L[i * n + i];
  }
}