#include <time.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

static void init_array(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{

    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double)n;

    for (int i = 0; i < n; i++)
    {
        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        v1[i] = ((i + 1) / fn) / 4.0;
        v2[i] = ((i + 1) / fn) / 6.0;
        y[i] = ((i + 1) / fn) / 8.0;
        z[i] = ((i + 1) / fn) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (int j = 0; j < n; j++)
            A[i * n + j] = (double)(i * j % n) / n;
    }
}

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

    // initialzize data on process 0
    if (rank == 0) {
        init_array(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
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