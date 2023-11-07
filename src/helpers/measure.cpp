#include <time.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
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
    init_array(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
    clock_gettime(CLOCK_MONOTONIC, &start);
    func(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // Calculate the elapsed time in seconds and nanoseconds
    elapsed_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;

    // write to output file
    outputFile << n << ";" << elapsed_time << ";" << functionName << "\n";

    
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