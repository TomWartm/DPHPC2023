#include <time.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "trisolv_init.h"
#include "gemver_init.h"

void measure_gemver(std::string functionName, void (*func)(int, double, double, double *, double *, double *, double *, double *, double *, double *, double *, double *), int n, std::ofstream &outputFile)
{

    // initialization

    double alpha;
    double beta;
    double *A = (double *)malloc((n * n) * sizeof(double));
    double *u1 = (double *)malloc((n) * sizeof(double));
    double *v1 = (double *)malloc((n) * sizeof(double));
    double *u2 = (double *)malloc((n) * sizeof(double));
    double *v2 = (double *)malloc((n) * sizeof(double));
    double *w = (double *)malloc((n) * sizeof(double));
    double *x = (double *)malloc((n) * sizeof(double));
    double *y = (double *)malloc((n) * sizeof(double));
    double *z = (double *)malloc((n) * sizeof(double));

    //////////////measure/////////////
    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_MONOTONIC, &start);
    init_gemver(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
    func(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // Calculate the elapsed time in seconds and nanoseconds
    elapsed_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    // write to output file
    outputFile << n << ";" << elapsed_time << ";" << functionName << std::endl;


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

void measure_trisolv(std::string functionName,void (*func)(int , double*, double*, double*), int n, std::ofstream &outputFile)
{
    double *L = (double *)malloc((n * n) * sizeof(double));
    double *x = (double *)malloc((n) * sizeof(double));
    double *b = (double *)malloc((n) * sizeof(double));

    //////////////measure/////////////
    struct timespec start, end;
    double elapsed_time;
    init_trisolv(n, L, x, b);
    clock_gettime(CLOCK_MONOTONIC, &start);
    func(n, L, x, b);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // Calculate the elapsed time in seconds and nanoseconds
    elapsed_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    // write to output file
    outputFile << n << ";" << elapsed_time << ";" << functionName << std::endl;

    // free memory
    free((void*)L);
    free((void*)x);
    free((void*)b);
}
