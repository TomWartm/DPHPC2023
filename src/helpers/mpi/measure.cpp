#include <time.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <mpi.h>
#include "../gemver_init.h"
#include "../trisolv_init.h"

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


    // MPI_Free_mem memory
    MPI_Free_mem((void *)A);
    MPI_Free_mem((void *)u1);
    MPI_Free_mem((void *)v1);
    MPI_Free_mem((void *)u2);
    MPI_Free_mem((void *)v2);
    MPI_Free_mem((void *)w);
    MPI_Free_mem((void *)x);
    MPI_Free_mem((void *)y);
    MPI_Free_mem((void *)z);
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

    // initialize data on all processes
    init_trisolv(n, L, x, b);

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &start);
    func(n, L, x, b);
    clock_gettime(CLOCK_MONOTONIC, &end);
    MPI_Barrier(MPI_COMM_WORLD);
    // Calculate the elapsed time in seconds and nanoseconds
    elapsed_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    // write to output file
    if (rank == 0) {
        outputFile << n << ";" << elapsed_time << ";" << functionName << std::endl;
    }

    // free memory
    MPI_Free_mem(L);
    MPI_Free_mem(x);
    MPI_Free_mem(b);
}