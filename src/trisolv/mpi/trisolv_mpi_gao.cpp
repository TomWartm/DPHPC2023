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
#include "../../helpers/mpi/util_gao.h"


double trisolv_mpi_gao(int size, int rank, int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
	if (N <= 512) return trisolv_naive(size, rank, N, A, x, b, init);
//	else if (N <= 8192 && (N / size) % 2 == 0) return trisolv_mpi_gao_double(size, rank, N, A, x, b, init);
	else return trisolv_mpi_gao_single(size, rank, N, A, x, b, init);
}

double trisolv_naive(int size, int rank, int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
	if (rank == 0) {
        A = new double[N * N];
        x = new double[N];
        b = new double[N];
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
        A_size = std_rows;
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

double trisolv_mpi_gao_any(int size, int rank, int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*), int block_size) {
    int rows, A_size;
    int std_rows = std::ceil(1.0 * N / size);
	int N_mod_std_rows = N % std_rows;
    if (rank == 0) {
        A_size = N;
        rows = std_rows;
    }
    else if (rank == size - 1 && N_mod_std_rows != 0) {
        A_size = std_rows;
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
    int sender, remain_rows, send_count, j_plus_k;
    double x_tmp;
    int rank_x_std_rows = rank * std_rows;
    
    for (int j = 0; j < N; j += send_count) {
        sender = j / std_rows;
        if (sender == size - 1 && N_mod_std_rows != 0) remain_rows = N_mod_std_rows;
        else remain_rows = std_rows;
        remain_rows -= j % std_rows;
        send_count = remain_rows >= block_size || remain_rows == 0 ? block_size: remain_rows;
        
        if (rank == sender) { //Compute small triangular matrix of size BLOCK_SIZE
        	for (int k = 0; k < send_count; ++k) {
        		j_plus_k = j + k;
        		x[j_plus_k] = b[j_plus_k] / A[(j_plus_k - rank_x_std_rows) + (j_plus_k) * A_size];
        		x_tmp = x[j_plus_k];
        		for (int l = 1; l < send_count; ++l) {
        			b[j + l] -= A[j + l - rank_x_std_rows + (j_plus_k) * A_size] * x_tmp;
        		}
        	}
        }
        
#ifdef TIME_BCAST
		if (rank == 0) b_start = std::chrono::high_resolution_clock::now();
       	MPI_Bcast(x + j, send_count, MPI_DOUBLE, sender, MPI_COMM_WORLD);
        if (rank == 0) {
        	b_end = std::chrono::high_resolution_clock::now();
        	bcast_dur = b_end - b_start;
        	bcast_time += bcast_dur.count();
        }
#else
		MPI_Bcast(x + j, send_count, MPI_DOUBLE, sender, MPI_COMM_WORLD);
#endif

        for (int k = 0; k < send_count; ++k) { //b - A * x
	        for (int i = 0; i < rows; ++i) {  		        	
    	        b[rank_x_std_rows + i] -= A[(j + k) * A_size + i] * x[j + k];
    	    }
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
        std::cout << block_size << "\t" << N << "\t" << diff.count()
#ifdef TIME_BCAST
        	<< "\t" << bcast_time << "\t" << bcast_time / diff.count() * 100 << "%"
#endif
        	<< "\n";
        return diff.count();
    }
    /*******************************************/
    return 0.0;
}
