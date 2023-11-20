#include <mpi.h>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <iomanip>

void init(int N, double* A, double* x, double* b, int first, int rows) {
	for (int j = 0; j < N; ++j) {
		for (int i = 0; i < rows; ++i) {
			A[j * N + i] = 1.0 * (i + first >= j);
		}
	}
	int k = 0;
	for (int i = 0; i < N; ++i) {
		k += i + 1;
		x[i] = 0.0;
		b[i] = k;
	}
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < N; ++j) {
			std::cout << A[j * N + i] << " ";
		}
		std::cout << "\n";
	}
}

int main(int argc, char** argv) {
	int NDEF = std::pow(2, POW);
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int std_rows = std::ceil(1.0 * NDEF / size);
	int rows = std_rows;
	if (rank == size - 1 && NDEF % rows != 0) {
		std::cout << "TRUE\n";
		rows = NDEF % std_rows;
	}
	std::cout << rank << "    " << rows << "\n";
	double *A = nullptr, *x = nullptr, *b = nullptr;
	MPI_Alloc_mem(rows * NDEF * sizeof(double), MPI_INFO_NULL, &A);
	MPI_Alloc_mem(NDEF * sizeof(double), MPI_INFO_NULL, &x);
	MPI_Alloc_mem(NDEF * sizeof(double), MPI_INFO_NULL, &b);
	init(NDEF, A, x, b, std_rows * rank, rows);
	
	
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
#ifdef PRINTX
		std::cout << "x = [";
		for (int i = 0; i < NDEF; ++i) std::cout << x[i] << " ";
		std::cout << "]\n";
#endif
		const std::chrono::duration<double> diff = end - start;
		std::cout << std::fixed << std::setprecision(9) << std::left;
        std::cout << "Time: " << diff.count() << '\n';
	}
		
	MPI_Free_mem(A);
	MPI_Free_mem(x);
	MPI_Free_mem(b);
	MPI_Finalize();
	return 0;
}
