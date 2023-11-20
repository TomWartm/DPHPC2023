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
			A[j * rows + i] = 1.0 * (i + first >= j);
		}
	}
	int k = 0;
	for (int i = 0; i < N; ++i) {
		k += i + 1;
		x[i] = 0.0;
		b[i] = k;
	}
	/*
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < N; ++j) {
			std::cout << A[j * rows + i] << " ";
		}
		std::cout << "\n";
	}*/
}

void benchmark(int size, int rank) {
	double *A = nullptr, *x = nullptr, *b = nullptr;
	for (int i = 6; i <= POW; ++i) {
		int std_rows = std::ceil(1.0 * NDEF / size);
		int rows = std_rows;
		if (rank == size - 1 && NDEF % rows != 0) rows = NDEF % std_rows;

		MPI_Alloc_mem(rows * NDEF * sizeof(double), MPI_INFO_NULL, &A);	
		MPI_Alloc_mem(NDEF * sizeof(double), MPI_INFO_NULL, &x);
		MPI_Alloc_mem(NDEF * sizeof(double), MPI_INFO_NULL, &b);
		init(NDEF, A, x, b, std_rows * rank, rows);
		
		MPI_Barrier(MPI_COMM_WORLD);
		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		if (rank == 0) {
			std::cout << "N = " << NDEF << "\n#CPUS = " << size << "\n";
			start = std::chrono::high_resolution_clock::now();
		}
		for (int j = 0; j < NDEF; ++j) {
			if (rank == j / std_rows) x[j] = b[j] / A[(j - rank * std_rows) * rows + j];
			MPI_Bcast(x + j, 1, MPI_DOUBLE, j / std_rows, MPI_COMM_WORLD);
			for (int i = 0; i < rows; ++i) {
				b[rank * std_rows + i] -= A[j * rows + i] * x[j];
			}
		}
		
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
	}
}

int main(int argc, char** argv) {
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	for (int repeat = 0; repeat < REPEAT; ++repeat) {
		benchmark(size, rank);
	}
	
	MPI_Finalize();
	return 0;
}
