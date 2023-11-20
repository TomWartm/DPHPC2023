#include "trisolv_openmp.h"

//v0 represents the baseline function, used as a placeholder
void trisolv_openmp_v0(int n, double* L, double* x, double* b){
    for (int i = 0; i < n; i++)
    {
        x[i] = b[i];
        for (int j = 0; j <i; j++)
        {
            double tmp = -L[i * n + j] * x[j];
            x[i] = x[i] +tmp;
        }
        x[i] = x[i] / L[i * n + i];
    }
}