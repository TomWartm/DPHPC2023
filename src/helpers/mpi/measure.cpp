#include <time.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <mpi.h>
#include "../gemver_init.h"
#include "../trisolv_init.h"

void measure_gemver_mpi(std::string functionName, void (*func)(int, double *, double *, double *), int n, std::ofstream &outputFile)
{   
    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // initialization
    double *A_result, *x_result, *w_result;
    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A_result);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_result);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &w_result);

    //////////////measure/////////////
    struct timespec start, end;
    double elapsed_time;

       
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Broadcast data to all other 
    MPI_Barrier(MPI_COMM_WORLD);
    func(n, A_result, x_result, w_result);
    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (rank == 0){
        // Calculate the elapsed time in seconds and nanoseconds
        elapsed_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

        // write to output file
        outputFile << n << ";" << elapsed_time << ";" << functionName << "\n";
    }


    // free memory
    free((void *)A_result);
    free((void *)x_result);
    free((void *)w_result);
    
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