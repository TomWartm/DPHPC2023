#include <mpi.h>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <iomanip>

void init(int N, double* A, double* x, double* b) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			A[i * N + j] = 1.0 * (j <= i);
		}
	}
	int k = 0;
	for (int i = 0; i < N; ++i) {
		k += i + 1;
		x[i] = 0.0;
		b[i] = k;
	}
}

#define N 16384

int main(int argc, char** argv) {
	int size, rank;
	double *A, *x, *b;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int std_rows = std::ceil(1.0 * N / size);
	int rows = std_rows;
	if (rank == size - 1 && N % rows != 0) rows = N % std_rows;
  
	if (rank == 0) {
		A = (double*)malloc(N * N * sizeof(double));
		x = (double*)malloc(N * sizeof(double));
		b = (double*)malloc(N * sizeof(double));
		init(N, A, x, b);
		for (int i = 1; i < size - 1; ++i) {
			MPI_Send(A + i * std_rows * N, std_rows * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
			std::cout << rank << " sending " << std_rows * N << " elements to " << i << "\n";
			MPI_Send(x, N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
			MPI_Send(b, N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
		}
		int count = N % std_rows == 0 ? std_rows: N % std_rows;
		MPI_Send(A + (size - 1) * std_rows * N, count * N, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD);
		std::cout << rank << " sending " << count * N << " elements to " << size - 1 << "\n";
		MPI_Send(x, N, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD);
		MPI_Send(b, N, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD);
	}
	else {
  	A = (double*)malloc(rows * N * sizeof(double));
  	x = (double*)malloc(N * rows);
  	b = (double*)malloc(N * rows);
		MPI_Recv(A, rows * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		std::cout << rank << " received " << rows * N << " elements\n";
		MPI_Recv(x, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(b, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	if (rank == 0) {
		start = std::chrono::high_resolution_clock::now();
	}
	MPI_Barrier(MPI_COMM_WORLD);
	for (int j = 0; j < N; ++j) {
		if (rank == j / std_rows) x[j] = b[j] / A[(j - rank * std_rows) * N + j];
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(x + j, 1, MPI_DOUBLE, j / std_rows, MPI_COMM_WORLD);
#ifdef OMP
#pragma omp for
#endif
		for (int i = 0; i < rows; ++i) {
#ifdef OMP
			std::cout << "THREAD " << omp_get_thread_num() << "\n";
#endif
			b[rank * std_rows + i] -= A[i * N + j] * x[j];
		}
	}
	
	if (rank == 0) {
		end = std::chrono::high_resolution_clock::now();
		std::cout << "x = [";
		for (int i = 0; i < N; ++i) std::cout << x[i] << " ";
		std::cout << "]\n";
		const std::chrono::duration<double> diff = end - start;
		std::cout << std::fixed << std::setprecision(9) << std::left;
        std::cout << "Time: " << diff.count() << '\n';
	}
		
	free(A);
	free(x);
	free(b);
	MPI_Finalize();
	return 0;
}
