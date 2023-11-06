#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include "../src/gemver/gemver_baseline.h"
#include "../src/gemver/gemver_mpi.h"

static void init_array(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z) {

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
void init_array_portion(int n, int rank, int num_procs, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z) {
    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double)n;
    int elements_per_proc = n / num_procs;
    int start = rank * elements_per_proc;
    int end = (rank + 1) * elements_per_proc;

    for (int i = start; i < end; i++) {
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

TEST(gemverTest, kernel_gemver){
    
    int n = 10;
    double alpha;
    double beta;
    double *A, *A_baseline = (double *)malloc((n * n) * sizeof(double));
    double *u1, *u1_baseline = (double *)malloc((n) * sizeof(double));
    double *v1, *v1_baseline = (double *)malloc((n) * sizeof(double));
    double *u2, *u2_baseline = (double *)malloc((n) * sizeof(double));
    double *v2, *v2_baseline = (double *)malloc((n) * sizeof(double));
    double *w , *w_baseline= (double *)malloc((n) * sizeof(double));
    double *x , *x_baseline= (double *)malloc((n) * sizeof(double));
    double *y , *y_baseline= (double *)malloc((n) * sizeof(double));
    double *z , *z_baseline= (double *)malloc((n) * sizeof(double));

    init_array(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
    init_array(n, &alpha, &beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);

    kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
    gemver_baseline(n, alpha, beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);

    for (int i = 0; i < n * n; i++) {
        ASSERT_NEAR(A[i], A_baseline[i], 1e-6);
    }
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(x[i], x_baseline[i], 1e-6);
    }
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(w[i], w_baseline[i], 1e-6);
    }
    //ASSERT_EQ(1,1); // TODO: check results

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

    free((void *)A_baseline);
    free((void *)u1_baseline);
    free((void *)v1_baseline);
    free((void *)u2_baseline);
    free((void *)v2_baseline);
    free((void *)w_baseline);
    free((void *)x_baseline);
    free((void *)y_baseline);
    free((void *)z_baseline);
}
/// @brief 
/// @param  
/// @param  
TEST(gemverTest, gemver_mpi_1){
    
    int rank, num_procs;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 10;
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
    double *A_baseline;
    double *u1_baseline;
    double *v1_baseline;
    double *u2_baseline;
    double *v2_baseline;
    double *w_baseline;
    double *x_baseline;
    double *y_baseline;
    double *z_baseline;
    // Determine the number of elements to allocate for each process
    int elements_per_proc = n / num_procs;
    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &u1);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v1);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &u2);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v2);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &w);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &y);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &z);


    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &u1_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v1_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &u2_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v2_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &w_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &y_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &z_baseline);
    
    if (rank == 0){
        // Allocate memory




        // Initialize memory
        init_array(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
        init_array(n, &alpha, &beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);
        kernel_gemver(n, alpha, beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);
        

    }
        // TODO: compute gemver in paralell

    // Broadcast the variables to all other processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(A, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(z, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // if(rank == 0){
    //     gemver_mpi_1(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z); // just stupidly compute same in all processes
    // }
    double *A_result;
    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A_result);
    MPI_Reduce(A,A_result,n*n, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (rank == 0){

        // check results in process 0
        for (int i = 0; i < n * n; i++) {
        ASSERT_NEAR(A[i], A_baseline[i], 1e-6);
        }
        for (int i = 0; i < n; i++) {
            ASSERT_NEAR(x[i], x_baseline[i], 1e-6);
        }
        for (int i = 0; i < n; i++) {
            ASSERT_NEAR(w[i], w_baseline[i], 1e-6);
        }

        // Free the allocated memory
        MPI_Free_mem(A);
        MPI_Free_mem(u1);
        MPI_Free_mem(v1);
        MPI_Free_mem(u2);
        MPI_Free_mem(v2);
        MPI_Free_mem(w);
        MPI_Free_mem(x);
        MPI_Free_mem(y);
        MPI_Free_mem(z);
        MPI_Free_mem(A_baseline);
        MPI_Free_mem(u1_baseline);
        MPI_Free_mem(v1_baseline);
        MPI_Free_mem(u2_baseline);
        MPI_Free_mem(v2_baseline);
        MPI_Free_mem(w_baseline);
        MPI_Free_mem(x_baseline);
        MPI_Free_mem(y_baseline);
        MPI_Free_mem(z_baseline);


    }
    
    



    
    
}








int main(int argc, char* argv[]) {
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}