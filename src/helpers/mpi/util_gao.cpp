#include <mpi.h>
#include <cmath>
#include "util_gao.h"

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

void init_colMaj(int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
	MPI_Free_mem(A);
	
	int size, rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	int std_rows = std::ceil(1.0 * N / size);
    
	if (rank == 0) {
        double* tmp;
        MPI_Alloc_mem(N * N * sizeof(double), MPI_INFO_NULL, &tmp);
        MPI_Alloc_mem(N * N * sizeof(double), MPI_INFO_NULL, &A);
        init(N, tmp, x, b);
        rowMaj_to_colMaj(N, tmp, A);
	    MPI_Free_mem(tmp);

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
        MPI_Alloc_mem(std_rows * N * sizeof(double), MPI_INFO_NULL, &A);
        MPI_Recv(A, N * std_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   		MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   		MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}
