#include <stdlib.h>
#include <cstring>
#include <mpi.h>
#include <cmath>
#include "trisolv_init.h"

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

/*
* For Gao's mpi approach
*/
void init_colMaj(int N, double* A, double* x, double* b) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
	int std_rows = std::ceil(1.0 * N / size);
    
	if (rank == 0) {
        double* tmp = new double[N * N];
        init_trisolv(N, tmp, x, b);
        rowMaj_to_colMaj(N, tmp, A);
        delete[] tmp;

       	double* partial_A = new double[std_rows * N];
        for (int n = 1; n < size; ++n) {
			get_partial(size, n, N, A, partial_A);
			MPI_Send(partial_A, N * std_rows, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
        }
        delete[] partial_A;
   		MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   		MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Recv(A, N * std_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   		MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   		MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void rowMaj_to_colMaj(int N, double* source, double* target) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            target[j * N + i] = source[i * N + j];
        }
    }
}

void get_partial(int size, int target_rank, int N, double* source, double* target) {
	int std_rows = std::ceil(1.0 * N / size);
	int starting_row = target_rank * std_rows;
    
    for (int j = 0; j < N; ++j) {
    	for (int i = 0; i < std_rows; ++i) {
    		if (i + starting_row >= N) target[j * std_rows + i] = -1;
    		else target[j * std_rows + i] = source[j * N + i + starting_row];
    	}
    }
}