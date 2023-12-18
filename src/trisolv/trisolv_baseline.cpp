#include <cblas.h>

void trisolv_baseline(int n, double* L, double* x, double* b)
{
  for (int i = 0; i < n; i++)
  {
      x[i] = b[i];
      for (int j = 0; j < i; j++)
      {
          double tmp = -L[i * n + j] * x[j];
          x[i] = x[i] + tmp;
      }
      x[i] = x[i] / L[i * n + i];
  }
}

void trisolv_openblas(int n, double* L, double* x, double* b) {
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, L, n, b, 1);
    for (int i = 0; i < n; ++i) {
        x[i] = b[i];
    }
}