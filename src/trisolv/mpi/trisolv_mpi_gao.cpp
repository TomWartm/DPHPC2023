//
// Created by gao on 27.11.23.
//
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
#include "trisolv_mpi_gao.h"

void rowMaj_to_colMaj(int N, double* source, double* target) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            target[j * N + i] = source[i * N + j];
        }
    }
}

void init_colMaj(int size, int rank, int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
	int std_rows = std::ceil(1.0 * N / size);
    int rows;

    if (rank == size - 1 && N % std_rows != 0) rows = N % std_rows;
    else rows = std_rows;
    
	if (rank == 0) {
        double* tmp = new double[N * N];
        A = new double[N * N];
        x = new double[N * sizeof(double)];
        b = new double[N * sizeof(double)];
        init(N, tmp, x, b);
        rowMaj_to_colMaj(N, tmp, A);
        delete[] tmp;

        for (int n = 1; n < size - 1; ++n) {
            for (int j = 0; j < N; ++j) {
                MPI_Send(A + n * std_rows + j * N, std_rows, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
            }
            MPI_Send(x, N, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
            MPI_Send(b, N, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
        }
        if (size > 1) {   //special treatment of the last rank
            int tmp_rows = std_rows;
            if (N % std_rows != 0) tmp_rows = N % std_rows;
            for (int j = 0; j < N; ++j) {
                MPI_Send(A + (size - 1) * std_rows + j * N, tmp_rows, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD);
            }
            MPI_Send(x, N, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD);
            MPI_Send(b, N, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD);
        }
    }
    else {
        A = new double[rows * N];
        x = new double[N * sizeof(double)];
        b = new double[N * sizeof(double)];
        for (int j = 0; j < N; ++j) {
            MPI_Recv(A + j * rows, rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Recv(x, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

double trisolv_mpi_gao(int size, int rank, int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
	if (N <= 512) return trisolv_naive(size, rank, N, A, x, b, init);
	else if (N <= 8192 && (N / size) % 2 == 0) return trisolv_mpi_gao_double(size, rank, N, A, x, b, init);
	else return trisolv_mpi_gao_single(size, rank, N, A, x, b, init);
}

double trisolv_naive(int size, int rank, int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
	if (rank == 0) {
        A = new double[N * N];
        x = new double[N * sizeof(double)];
        b = new double[N * sizeof(double)];
        init(N, A, x, b);
		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < N; i++) {
			x[i] = b[i];
			int i_x_N = i * N;
			for (int j = 0; j < i; j++) {
				x[i] -= A[i_x_N + j] * x[j];
			}
			x[i] /= A[i_x_N + i];
		}
        end = std::chrono::high_resolution_clock::now();
		const std::chrono::duration<double> diff = end - start;
#ifdef PRINT_X
        std::cout << "x = [";
		for (int i = 0; i < N; ++i) std::cout << x[i] << " ";
		std::cout << "]\n";
#endif
		std::cout << std::fixed << std::setprecision(9) << std::left;
		std::cout << "naive" << "\t" << N << "\t" << diff.count() << "\n";
		return diff.count();
	}
	else return 0;
}

double trisolv_mpi_gao_single(int size, int rank, int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
    int std_rows = std::ceil(1.0 * N / size);
    int rows, A_size;

    if (rank == 0) {
        A_size = N;
        rows = std_rows;
    }
    else if (rank == size - 1 && N % std_rows != 0) {
        A_size = N % std_rows;
        rows = N % std_rows;
    }
    else {
        A_size = std_rows;
        rows = std_rows;
    }


    /****************INITIALIZATION******************/
    init_colMaj(size, rank, N, A, x, b, init);
    /************************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    /****************START TIMER******************/
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
#ifdef TIME_BCAST
	std::chrono::time_point<std::chrono::high_resolution_clock> b_start, b_end;
    std::chrono::duration<double> bcast_dur;
    double bcast_time = 0;
#endif
    if (rank == 0) {
        start = std::chrono::high_resolution_clock::now();
    }
    /*********************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    /****************COMPUTATION******************/
    for (int j = 0; j < N; ++j) {
        int rank_x_std_rows = rank * std_rows;
        int j_x_A_size = j * A_size;
        if (rank == j / std_rows) x[j] = b[j] / A[(j - rank_x_std_rows) + j * A_size];
#ifdef TIME_BCAST
		if (rank == 0) b_start = std::chrono::high_resolution_clock::now();
#endif
        MPI_Bcast(x + j, 1, MPI_DOUBLE, j / std_rows, MPI_COMM_WORLD);
#ifdef TIME_BCAST
        if (rank == 0) {
        	b_end = std::chrono::high_resolution_clock::now();
        	bcast_dur = b_end - b_start;
        	bcast_time += bcast_dur.count();
        }
#endif
        for (int i = 0; i < rows; ++i) {
            b[rank_x_std_rows + i] -= A[j_x_A_size + i] * x[j];
        }
    }
    /*********************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    /****************END TIMER******************/
    if (rank == 0) {
        end = std::chrono::high_resolution_clock::now();
#ifdef PRINT_X
        std::cout << "x = [";
		for (int i = 0; i < N; ++i) std::cout << x[i] << " ";
		std::cout << "]\n";
#endif
        const std::chrono::duration<double> diff = end - start;
        std::cout << std::fixed << std::setprecision(9) << std::left;
        std::cout << "single" << "\t" << N << "\t" << diff.count()
#ifdef TIME_BCAST
        	<< "\t" << bcast_time << "\t" << bcast_time / diff.count() * 100 << "%"
#endif
        	<< "\n";
        return diff.count();
    }
    /*******************************************/
    return 0.0;
}

double trisolv_mpi_gao_double(int size, int rank, int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
    int std_rows = std::ceil(1.0 * N / size);
    int rows, A_size;

    if (rank == 0) {
        A_size = N;
        rows = std_rows;
    }
    else if (rank == size - 1 && N % std_rows != 0) {
        A_size = N % std_rows;
        rows = N % std_rows;
    }
    else {
        A_size = std_rows;
        rows = std_rows;
    }


    /****************INITIALIZATION******************/
    init_colMaj(size, rank, N, A, x, b, init);
    /************************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    /****************START TIMER******************/
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
#ifdef TIME_BCAST
	std::chrono::time_point<std::chrono::high_resolution_clock> b_start, b_end;
    std::chrono::duration<double> bcast_dur;
    double bcast_time = 0;
#endif
    if (rank == 0) {
        start = std::chrono::high_resolution_clock::now();
    }
    /*********************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    /****************COMPUTATION******************/
    for (int j = 0; j < N; j += 2) {
        int rank_x_std_rows = rank * std_rows;
        if (rank == j / std_rows) {
        	x[j] = b[j] / A[(j - rank_x_std_rows) + j * A_size];
        	b[j + 1] -= A[(j + 1 - rank_x_std_rows) + j * A_size] * x[j];
        	x[j + 1] = b[j + 1] / A[(j + 1 - rank_x_std_rows) + (j + 1) * A_size];
        }
#ifdef TIME_BCAST
		if (rank == 0) b_start = std::chrono::high_resolution_clock::now();
#endif
        MPI_Bcast(x + j, 2, MPI_DOUBLE, j / std_rows, MPI_COMM_WORLD);
#ifdef TIME_BCAST
        if (rank == 0) {
        	b_end = std::chrono::high_resolution_clock::now();
        	bcast_dur = b_end - b_start;
        	bcast_time += bcast_dur.count();
        }
#endif
        int j0_x_A_size = j * A_size;
        int j1_x_A_size = (j + 1) * A_size;
        for (int i = 0; i < rows; ++i) {
            b[rank_x_std_rows + i] -= A[j0_x_A_size + i] * x[j] + A[j1_x_A_size + i] * x[j + 1];
        }
    }
    /*********************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    /****************END TIMER******************/
    if (rank == 0) {
        end = std::chrono::high_resolution_clock::now();
#ifdef PRINT_X
        std::cout << "x = [";
		for (int i = 0; i < N; ++i) std::cout << x[i] << " ";
		std::cout << "]\n";
#endif
        const std::chrono::duration<double> diff = end - start;
        std::cout << std::fixed << std::setprecision(9) << std::left;
        std::cout << "double" << "\t" << N << "\t" << diff.count()
#ifdef TIME_BCAST
        	<< "\t" << bcast_time << "\t" << bcast_time / diff.count() * 100 << "%"
#endif
        	<< "\n";
        return diff.count();
    }
    /*******************************************/
    return 0.0;
}
