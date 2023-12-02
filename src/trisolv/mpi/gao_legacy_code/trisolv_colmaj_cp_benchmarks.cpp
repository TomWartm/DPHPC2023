#include <mpi.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>

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

double run_mpi (int size, int rank, int NDEF) {
    //std::cout << rank << "|" << N << "\n";
    double time = 0;
    double *A, *x, *b;
    int std_rows = std::ceil(1.0 * NDEF / size);
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
#ifdef PRINTX
        std::cout << "x = [";
		for (int i = 0; i < NDEF; ++i) std::cout << x[i] << " ";
		std::cout << "]\n";
#endif
        const std::chrono::duration<double> diff = end - start;
        //std::cout << std::fixed << std::setprecision(9) << std::left;
        //std::cout << "Time: " << diff.count() << '\n';
        time = diff.count();
    }
    /*******************************************/

    delete A;
    delete x;
    delete b;
    return time;
}

double run_baseline (int size, int rank, int NDEF) {
    //std::cout << rank << "|" << N << "\n";
    double time = 0;
    double *A, *x, *b;
    if (rank == 0) {
        A = new double[NDEF * NDEF];
        x = new double[NDEF * sizeof(double)];
        b = new double[NDEF * sizeof(double)];
        init(NDEF, A, x, b);
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        start = std::chrono::high_resolution_clock::now();

#pragma scop
        for (int i = 0; i < NDEF; i++)
        {
            x[i] = b[i];
            for (int j = 0; j < i; j++)
                x[i] -= A[i * NDEF + j] * x[j];
            x[i] = x[i] / A[i * NDEF + i];
        }
#pragma endscop

        end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> diff = end - start;
        time = diff.count();
        delete A;
        delete x;
        delete b;
    }
    return time;
}

template <class F>
void benchmark(int size, int rank, F func) {
	for (int i = 6; i <= POW; ++i) {
        int N = std::pow(2, i);
        std::vector<double> time;
        for (int j = 0; j < REPEAT; ++j) {
            double tmp = func(size, rank, N);
            if (rank == 0)
                time.push_back(tmp);
        }
        if (rank == 0) {
            std::sort(time.begin(), time.end());
            std::cout << N << "\t";
            std::cout << std::fixed << std::setprecision(9) << std::left;
            std::cout << time[REPEAT / 2] << '\n';
        }
	}
}

int main(int argc, char** argv) {
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "=====MPI=====\n";
    benchmark(size, rank, run_mpi);
    if (rank == 0) std::cout << "\n==BASELINE===\n";
    benchmark(size, rank, run_baseline);
	
	MPI_Finalize();
	return 0;
}
