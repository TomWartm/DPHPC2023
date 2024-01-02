#ifndef TRISOLV_BASELINE_H
#define TRISOLV_BASELINE_H

// Function to perform the TRISOLV operation
void trisolv_baseline(int n, double* L, double* x, double* b);
void trisolv_openblas(int n, double* L, double* x, double* b);

// You can also declare any other necessary constants, data structures, or functions here

#endif // TRISOLV_BASELINE_H