#include <mpi.h>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <iomanip>

#define PADDING 16

void init(int N, double* A, double* x, double* b, int first, int rows) {
	for (int j = 0; j < N; ++j) {
		for (int i = 0; i < rows; ++i) {
			A[j * rows + i] = 1.0 * (i + first >= j);
		}
	}
	int k = 0;
	for (int i = 0; i < N * PADDING; ++i) {
		k += (i / 16 + 1) * (i % 16 == 0);
		x[i] = 0.0;
		b[i] = k * (i % 16 == 0);
	}
	
	//for (int i = 0; i < rows; ++i) {
/*	if (first == 0)
		for (int j = 0; j < N; ++j) {
			std::cout << x[j * PADDING] << "   " << b[j * PADDING] << "\n";
			//std::cout << A[j * rows + i] << " ";
		}*/
	//	std::cout << "\n";
	//}
}

int main(int argc, char** argv) {
	int NDEF = std::pow(2, POW);
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int std_rows = std::ceil(1.0 * NDEF / size);
	int rows = std_rows;
	if (rank == size - 1 && NDEF % rows != 0) rows = NDEF % std_rows;
	double *A = nullptr, *x = nullptr, *b = nullptr;
	MPI_Alloc_mem(rows * NDEF * sizeof(double), MPI_INFO_NULL, &A);
	MPI_Alloc_mem(NDEF * sizeof(double) * PADDING, MPI_INFO_NULL, &x);
	MPI_Alloc_mem(NDEF * sizeof(double) * PADDING, MPI_INFO_NULL, &b);
	init(NDEF, A, x, b, std_rows * rank, rows);
	
	MPI_Barrier(MPI_COMM_WORLD);
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	if (rank == 0) {
		std::cout << "N = " << NDEF << "\n#CPUS = " << size << "\n";
		std::cout << "THREAD NUM: " << omp_get_thread_num << "\n";
		start = std::chrono::high_resolution_clock::now();
	}
	for (int j = 0; j < NDEF; ++j) {
		if (rank == j / std_rows) x[j * PADDING] = b[j * PADDING] / A[(j - rank * std_rows) * rows + j];
		MPI_Bcast(x + j * PADDING, 1, MPI_DOUBLE, j / std_rows, MPI_COMM_WORLD);
		#pragma omp parallel for
		for (int i = 0; i < rows; ++i) {

			b[(rank * std_rows + i) * PADDING] -= A[j * rows + i] * x[j * PADDING];
		}
	}
	
	if (rank == 0) {
		end = std::chrono::high_resolution_clock::now();
#ifdef PRINTX
		std::cout << "x = [";
		for (int i = 0; i < NDEF; ++i) std::cout << x[i * PADDING] << " ";
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
