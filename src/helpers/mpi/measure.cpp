#include <time.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <mpi.h>
#include <chrono>
#include <iomanip>
#include "../gemver_init.h"
#include "../trisolv_init.h"
#include "../../trisolv/mpi/trisolv_mpi_gao.h"
#include "../../trisolv/trisolv_baseline.h"

void measure_gemver_mpi(std::string functionName, void (*func)(int, double, double, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *), int n, std::ofstream &outputFile)
{   
    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // initialization
    double alpha;
    double beta;
    double *A;
    double *u1;
    double *v1;
    double *u2;
    double *v2;
    double *w;
    double *x;
    double *y;
    double *z;

    double *A_result, *x_result, *w_result;

    // allocate memory on each process
    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &u1);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v1);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &u2);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v2);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &w);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &y);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &z);

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A_result);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_result);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &w_result);
    //////////////measure/////////////
    struct timespec start, end;
    double elapsed_time;

    // initialize data on process 0
    if (rank == 0) {
        init_gemver(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
    }
   
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Broadcast data to all other 
    MPI_Barrier(MPI_COMM_WORLD);
    func(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z, A_result, x_result, w_result);
    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (rank == 0){
        // Calculate the elapsed time in seconds and nanoseconds
        elapsed_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

        // write to output file
        outputFile << n << ";" << elapsed_time << ";" << functionName << "\n";
    }


    // free memory
    free((void *)A);
    free((void *)u1);
    free((void *)v1);
    free((void *)u2);
    free((void *)v2);
    free((void *)w);
    free((void *)x);
    free((void *)y);
    free((void *)z);
}


void measure_trisolv_mpi(std::string functionName,void (*func)(int , double*, double*, double*), int n, std::ofstream &outputFile)
{
    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double *L;
    double *x;
    double *b;

    MPI_Alloc_mem((n * n) * sizeof(double), MPI_INFO_NULL, &L);
    MPI_Alloc_mem((n) * sizeof(double), MPI_INFO_NULL, &x);
    MPI_Alloc_mem((n) * sizeof(double), MPI_INFO_NULL, &b);

    //////////////measure/////////////
    struct timespec start, end;
    double elapsed_time;

    // initialize data on process 0
    if (rank == 0) {
        init_trisolv(n, L, x, b);
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    // Broadcast data to all other
    MPI_Barrier(MPI_COMM_WORLD);
    func(n, L, x, b);
    clock_gettime(CLOCK_MONOTONIC, &end);
    MPI_Barrier(MPI_COMM_WORLD);
    // Calculate the elapsed time in seconds and nanoseconds
    elapsed_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    // write to output file
    outputFile << n << ";" << elapsed_time << ";" << functionName << std::endl;

    // free memory
    free((void*)L);
    free((void*)x);
    free((void*)b);
}

void measure_trisolv_baseline(int n, std::ofstream &outputFile) {
	int size, rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (rank == 0) {
	    double *A = new double[n * n];
    	double *x = new double[n];
	    double *b = new double[n];
    	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    	
	    init_trisolv(n, A, x, b);    	
		
	    start = std::chrono::high_resolution_clock::now();
    	trisolv_baseline(n, A, x, b);
		end = std::chrono::high_resolution_clock::now();
		const std::chrono::duration<double> time = end - start;
    	outputFile << n << ";" << time.count() << ";" << "trisolv_mpi_baseline" << std::endl;
    	std::cout << std::fixed << std::setprecision(9) << std::left;
		std::cout << "bline" << "\t" << n << "\t" << time.count() << "\n";
   	    if (A) delete[] A;
	    if (x) delete[] x;
	    if (b) delete[] b;
    }
}

double measure_trisolv_mpi(int n,
						std::ofstream &outputFile,
						double (*solver)(int, int, int, double*&, double*&, double*&, void (*)(int, double*, double*, double*)),
						const std::string& name) {
	int size, rank;
    double *A = nullptr, *x = nullptr, *b = nullptr;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time = solver(size, rank, n, A, x, b, lowertriangular_trisolv);
    if (rank == 0) outputFile << n << ";" << time << ";" << name << std::endl;

    if (A) delete[] A;
    if (x) delete[] x;
    if (b) delete[] b;
    return time;
}

double measure_trisolv_mpi_block(int n,
						std::ofstream &outputFile,
						double (*solver)(int, int, int, double*&, double*&, double*&, void (*)(int, double*, double*, double*), int),
						const std::string& name,
						int block_size) {
	int size, rank;
    double *A = nullptr, *x = nullptr, *b = nullptr;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time = solver(size, rank, n, A, x, b, lowertriangular_trisolv, block_size);
    if (rank == 0) outputFile << n << ";" << time << ";" << name << std::endl;

    if (A) delete[] A;
    if (x) delete[] x;
    if (b) delete[] b;
    return time;
}

