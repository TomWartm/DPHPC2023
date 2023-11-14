#include <mpi.h>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <iomanip>

void init(int N, double* A, double* x, double* b) {
	for (int j = 0; j < N; ++j) {
		for (int i = 0; i < N; ++i) {
			A[j * N + i] = 1.0 * (i >= j);
		}
	}
	int k = 0;
	for (int i = 0; i < N; ++i) {
		k += i + 1;
		x[i] = 0.0;
		b[i] = k;
	}
}

int main(int argc, char** argv) {
	int size, rank;
	double *A, *x, *b;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	for (int i = 6; i <= POW; ++i) {
		int NDEF = std::pow(2, i);
		int std_rows = std::ceil(1.0 * NDEF / size);
		int rows = std_rows;
		if (rank == size - 1 && NDEF % rows != 0) rows = NDEF % std_rows;
		if (rank == 0) {
			A = (double*)malloc(NDEF * NDEF * sizeof(double));
			x = (double*)malloc(NDEF * sizeof(double));
			b = (double*)malloc(NDEF * sizeof(double));
			init(NDEF, A, x, b);
			for (int n = 1; n < size; ++n) {
				for (int j = 0; j < NDEF; ++j) {
					MPI_Send(A + n * std_rows + j * NDEF, std_rows, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
				}
				MPI_Send(x, NDEF, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
				MPI_Send(b, NDEF, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
			}		
		}
		else {
			A = (double*)malloc(std_rows * NDEF * sizeof(double));
			x = (double*)malloc(NDEF * sizeof(double));
			b = (double*)malloc(NDEF * sizeof(double));
			for (int j = 0; j < NDEF; ++j) {
				MPI_Recv(A + j * std_rows, std_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			MPI_Recv(x, NDEF, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(b, NDEF, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		if (rank == 0) {
			start = std::chrono::high_resolution_clock::now();
		}
		MPI_Barrier(MPI_COMM_WORLD);
		for (int j = 0; j < NDEF; ++j) {
			if (rank == j / std_rows) x[j] = b[j] / A[(j - rank * std_rows) * std_rows + j];
			MPI_Bcast(x + j, 1, MPI_DOUBLE, j / std_rows, MPI_COMM_WORLD);
			for (int i = 0; i < std_rows; ++i) {
				b[rank * std_rows + i] -= A[j * std_rows + i] * x[j];
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			end = std::chrono::high_resolution_clock::now();
			const std::chrono::duration<double> diff = end - start;
			std::cout << NDEF << "\t";
			std::cout << std::fixed << std::setprecision(9) << std::left;
			std::cout << diff.count() << '\n';
		}
			
		free(A);
		free(x);
		free(b);
	}
	
	
	MPI_Finalize();
	return 0;
}
