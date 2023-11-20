#include <stdlib.h>
#include <cstring>
void init_trisolv(int n, double* L, double* x, double* b){
    memset(L, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        x[i] = -999;
        b[i] = i ;
        for (int j = 0; j <= i; j++)
            L[i * n + j] = (double) (i + n - j + 1) * 2 / n;
    }
}

void identity_trisolv(int n, double* L, double* x, double* b){
    memset(L, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++){
        x[i] = 1.0;
        b[i] = 1.0;
        L[i * n + i] = 1.0;
    }
}

void random_trisolv(int n, double* L, double* x, double* b){
    srand(42);
    // Fill L with random values
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            if(j <= i){
                L[i*n + j] = (double)rand() / RAND_MAX; // Random value between 0 and 1
            } else {
                L[i*n + j] = 0.0; // Zero for upper triangular part
            }
        }
    }

    // Fill x and b with random values
    for(int i = 0; i < n; ++i){
        x[i] = (double)rand() / RAND_MAX;
        b[i] = (double)rand() / RAND_MAX;
    }
}


void lowertriangular_trisolv(int n, double* L, double* x, double* b){
    // Initialize the lower triangular matrix L
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j <= i; ++j) {
            L[i * n + j] = 1.0;
        }
        for(int j = i + 1; j < n; ++j) {
            L[i * n + j] = 0.0;
        }
    }

    // Initialize the vector x
    for(int i = 0; i < n; ++i) {
        x[i] = i + 1; // x will be 1, 2, 3, ..., n
    }

    // Initialize the vector b with triangular numbers
    b[0] = 1;
    for(int i = 1; i < n; ++i) {
        b[i] = b[i - 1] + (i + 1); // b[i] = 1, 3, 6, 10, ..., n*(n+1)/2
    }
}