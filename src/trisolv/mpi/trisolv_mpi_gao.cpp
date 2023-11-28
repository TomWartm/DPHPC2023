//
// Created by gao on 27.11.23.
//
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
//#include <thread>
#include "trisolv_mpi_gao.h"

void print(double* A, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[j * M + i] << " ";
        }
        std::cout << "\n";
    }
}

void rowMaj_to_colMaj(int N, double* source, double* target) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            target[j * N + i] = source[i * N + j];
        }
    }
    //print(target, N, N);
}

double trisolv_mpi_gao(int size, int rank, int NDEF, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
    int std_rows = std::ceil(1.0 * NDEF / size);
    int rows, A_size;

    if (rank == 0) {
//    	std::cout << "THREADS: " << std::thread::hardware_concurrency() << "\n";
        A_size = NDEF;
        rows = std_rows;
    }
    else if (rank == size - 1 && NDEF % std_rows != 0) {
        A_size = NDEF % std_rows;
        rows = NDEF % std_rows;
    }
    else {
        A_size = std_rows;
        rows = std_rows;
    }


    /****************INITIALIZATION******************/
    if (rank == 0) {
        double* tmp = new double[NDEF * NDEF];
        A = new double[NDEF * NDEF];
        x = new double[NDEF * sizeof(double)];
        b = new double[NDEF * sizeof(double)];
        init(NDEF, tmp, x, b);
        rowMaj_to_colMaj(NDEF, tmp, A);
        delete[] tmp;

        //print(A, NDEF, NDEF);

        for (int n = 1; n < size - 1; ++n) {
            for (int j = 0; j < NDEF; ++j) {
                MPI_Send(A + n * std_rows + j * NDEF, std_rows, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
            }
            MPI_Send(x, NDEF, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
            MPI_Send(b, NDEF, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
        }
        if (size > 1) {   //special treatment of the last rank
            int tmp_rows = std_rows;
            if (NDEF % std_rows != 0) tmp_rows = NDEF % std_rows;
            for (int j = 0; j < NDEF; ++j) {
                //for (int k = 0; k < tmp_rows; ++k) std::cout << "!" << A[(size - 1) * std_rows + j * NDEF + k] << " ";
                //std::cout << "\n";
                MPI_Send(A + (size - 1) * std_rows + j * NDEF, tmp_rows, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD);
            }
            MPI_Send(x, NDEF, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD);
            MPI_Send(b, NDEF, MPI_DOUBLE, size - 1, 1, MPI_COMM_WORLD);
        }
    }
    else {
        A = new double[rows * NDEF];
        x = new double[NDEF * sizeof(double)];
        b = new double[NDEF * sizeof(double)];
        for (int j = 0; j < NDEF; ++j) {
            MPI_Recv(A + j * rows, rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Recv(x, NDEF, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, NDEF, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /*if (rank == size - 1)*/ //print(A, rows, NDEF);
    }
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
    for (int j = 0; j < NDEF; ++j) {
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
#pragma omp parallel for
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
		for (int i = 0; i < NDEF; ++i) std::cout << x[i] << " ";
		std::cout << "]\n";
#endif
        const std::chrono::duration<double> diff = end - start;
        std::cout << std::fixed << std::setprecision(9) << std::left;
        std::cout << NDEF << "\t" << diff.count()
#ifdef TIME_BCAST
        	<< "\t" << bcast_time << "\t" << bcast_time / diff.count() * 100 << "%"
#endif
        	<< "\n";
        return diff.count();
    }
    /*******************************************/
    return 0.0;
}
