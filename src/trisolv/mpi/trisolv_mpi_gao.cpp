//
// Created by gao on 27.11.23.
//
#include "trisolv_mpi_gao.h"

template <class Init>
double trisolv_mpi_gao(int size, int rank, int NDEF, double* A, double* x, double* b, Init init) {
    int std_rows = std::ceil(1.0 * NDEF / size);
    if (NDEF % size != 0) {
        if (rank == 0) std::cout << "SIZE OF MATRIX IS NOT DIVISIBLE BY NUMBER OF CPUS\n";
        MPI_Finalize();
        return 1;
    }
    /*int rows = std_rows;
    if (rank == size - 1 && NDEF % rows != 0) rows = NDEF % std_rows;*/
    int rows;
    if (rank == 0) rows = NDEF;
    else rows = std_rows;


    /****************INITIALIZATION******************/
    if (rank == 0) {
        A = new double[NDEF * NDEF];
        x = new double[NDEF * sizeof(double)];
        b = new double[NDEF * sizeof(double)];
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
        A = new double[std_rows * NDEF];
        x = new double[NDEF * sizeof(double)];
        b = new double[NDEF * sizeof(double)];
        for (int j = 0; j < NDEF; ++j) {
            MPI_Recv(A + j * std_rows, std_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Recv(x, NDEF, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, NDEF, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    /************************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    /****************START TIMER******************/
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    if (rank == 0) {
        start = std::chrono::high_resolution_clock::now();
    }
    /*********************************************/

    MPI_Barrier(MPI_COMM_WORLD);

    /****************COMPUTATION******************/
    for (int j = 0; j < NDEF; ++j) {
        int rank_x_std_rows = rank * std_rows;
        int j_x_rows = j * rows;
        if (rank == j / std_rows) x[j] = b[j] / A[(j - rank_x_std_rows) * rows + j];
        MPI_Bcast(x + j, 1, MPI_DOUBLE, j / std_rows, MPI_COMM_WORLD);
        for (int i = 0; i < std_rows; ++i) {
            b[rank_x_std_rows + i] -= A[j_x_rows + i] * x[j];
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
#ifdef PRINT_TIME
        std::cout << std::fixed << std::setprecision(9) << std::left;
        std::cout << NDEF << "\t" << diff.count() << '\n';
#endif
        return diff.count();
    }
    /*******************************************/

    return 0.0;
}