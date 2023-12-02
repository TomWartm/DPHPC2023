#ifndef TRISOLV_INIT_H
#define TRISOLV_INIT_H

// Function to initialize the trisolv arrays
void init_trisolv(int n, double* L, double* x, double* b);
void identity_trisolv(int n, double* L, double* x, double* b);
void random_trisolv(int n, double* L, double* x, double* b);
void lowertriangular_trisolv(int n, double* L, double* x, double* b);
void init_colMaj(int N, double* A, double* x, double* b);
void rowMaj_to_colMaj(int N, double* source, double* target);
void get_partial(int size, int target_rank, int N, double* source, double* target);
// You can also declare any other necessary constants, data structures, or functions here

#endif // TRISOLV_INIT_H