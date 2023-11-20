#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include "../../../src/gemver/gemver_baseline.h"
#include "../../../src/helpers/gemver_init.h"
#include "../../../src/gemver/mpi/gemver_mpi.h"

/// @brief
/// @param
/// @param
TEST(gemverTest, SparseInitialization)
{
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


    // initialize variables to store the results in
    double *A_result, *x_result, *w_result;
    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A_result);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_result);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &w_result);


    if (rank == 0)
    {
        // initialize data on processer 0
        sparse_init_gemver(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
        sparse_init_gemver(n, &alpha, &beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);
        kernel_gemver(n, alpha, beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);
    }

    // compute gemver
    kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z); // just stupidly compute same in all processes

    if (rank == 0)
    {

        // check results in process 0
        for (int i = 0; i < n * n; i++)
        {
            EXPECT_NEAR(A_result[i], A_baseline[i], 1e-6);
        }
        for (int i = 0; i < n; i++)
        {
            EXPECT_NEAR(x_result[i], x_baseline[i], 1e-6);
        }
        for (int i = 0; i < n; i++)
        {
            EXPECT_NEAR(w_result[i], w_baseline[i], 1e-6);
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

        MPI_Free_mem(A_result);
        MPI_Free_mem(x_result);
        MPI_Free_mem(w_result);
    }
}

TEST(gemverTest, RandomInitialization){
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


    // initialize variables to store the results in
    double *A_result, *x_result, *w_result;
    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A_result);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_result);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &w_result);


    if (rank == 0){
        // initialize data on processer 0
        rand_init_gemver(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
        rand_init_gemver(n, &alpha, &beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);
        kernel_gemver(n, alpha, beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);
    }

    // compute gemver
    kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z); // just stupidly compute same in all processes

    if (rank == 0){

        // check results in process 0
        for (int i = 0; i < n * n; i++){
            EXPECT_NEAR(A_result[i], A_baseline[i], 1e-6);
        }
        for (int i = 0; i < n; i++){
            EXPECT_NEAR(x_result[i], x_baseline[i], 1e-6);
        }
        for (int i = 0; i < n; i++){
            EXPECT_NEAR(w_result[i], w_baseline[i], 1e-6);
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

        MPI_Free_mem(A_result);
        MPI_Free_mem(x_result);
        MPI_Free_mem(w_result);
    }
}

TEST(gemverTest, DifferentSizes){
    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<int> n_vec(5);
    n_vec = {0, 10, 100, 1000, 10000};

    for (auto n : n_vec){
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


        // initialize variables to store the results in
        double *A_result, *x_result, *w_result;
        MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A_result);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_result);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &w_result);


        if (rank == 0){
            // initialize data on processer 0
            init_gemver(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
            init_gemver(n, &alpha, &beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);
            kernel_gemver(n, alpha, beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);
        }

        // compute gemver
        kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z); // just stupidly compute same in all processes

        if (rank == 0){

            // check results in process 0
            for (int i = 0; i < n * n; i++){
                EXPECT_NEAR(A_result[i], A_baseline[i], 1e-6);
            }
            for (int i = 0; i < n; i++){
                EXPECT_NEAR(x_result[i], x_baseline[i], 1e-6);
            }
            for (int i = 0; i < n; i++){
                EXPECT_NEAR(w_result[i], w_baseline[i], 1e-6);
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

            MPI_Free_mem(A_result);
            MPI_Free_mem(x_result);
            MPI_Free_mem(w_result);
        }
    }
}

int main(int argc, char *argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}