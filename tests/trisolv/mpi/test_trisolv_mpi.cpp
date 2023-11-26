#include <gtest/gtest.h>
#include <mpi.h>
#include "../../../src/trisolv/trisolv_baseline.h"
#include "../../../src/helpers/trisolv_init.h"
#include "../../../src/trisolv/mpi/trisolv_mpi.h"

/// @brief
/// @param
/// @param
TEST(trisolvMPITest, IdentityInitialization) {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 3; // Example size

    double *L, *x, *b;
    double *L_baseline, *x_baseline, *b_baseline;

    // Allocate memory using MPI
    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b);

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b_baseline);

    if (rank == 0) {
        // Initialize matrices and vectors on process 0
        identity_trisolv(n, L, x, b);
        identity_trisolv(n, L_baseline, x_baseline, b_baseline);

        // Compute baseline solution
        trisolv_baseline(n, L_baseline, x_baseline, b_baseline);
    }

    // Compute trisolv using MPI
    trisolv_mpi_v0(n, L, x, b); // This function needs to be implemented

    if (rank == 0) {
        // Check results in process 0
        for (int i = 0; i < n * n; i++) {
            EXPECT_NEAR(L[i], L_baseline[i], 1e-6);
        }
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
            EXPECT_NEAR(b[i], b_baseline[i], 1e-6);
        }

        // Free the allocated memory
        MPI_Free_mem(L);
        MPI_Free_mem(x);
        MPI_Free_mem(b);
        MPI_Free_mem(L_baseline);
        MPI_Free_mem(x_baseline);
        MPI_Free_mem(b_baseline);
    }
}

TEST(trisolvMPITest, RandomInitialization) {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 5; // Example size

    double *L, *x, *b;
    double *L_baseline, *x_baseline, *b_baseline;

    // Allocate memory using MPI
    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b);

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b_baseline);

    if (rank == 0) {
        // Initialize matrices and vectors on process 0
        random_trisolv(n, L, x, b);
        random_trisolv(n, L_baseline, x_baseline, b_baseline);

        // Compute baseline solution
        trisolv_baseline(n, L_baseline, x_baseline, b_baseline);
    }

    // Compute trisolv using MPI
    trisolv_mpi_v0(n, L, x, b); // This function needs to be implemented

    if (rank == 0) {
        // Check results in process 0
        for (int i = 0; i < n * n; i++) {
            EXPECT_NEAR(L[i], L_baseline[i], 1e-6);
        }
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
            EXPECT_NEAR(b[i], b_baseline[i], 1e-6);
        }

        // Free the allocated memory
        MPI_Free_mem(L);
        MPI_Free_mem(x);
        MPI_Free_mem(b);
        MPI_Free_mem(L_baseline);
        MPI_Free_mem(x_baseline);
        MPI_Free_mem(b_baseline);
    }
}

TEST(trisolvMPITest, LTriangularInitialization) {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 100; // Example size

    double *L, *x, *b;
    double *L_baseline, *x_baseline, *b_baseline;

    // Allocate memory using MPI
    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b);

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b_baseline);

    if (rank == 0) {
        // Initialize matrices and vectors on process 0
        lowertriangular_trisolv(n, L, x, b);
        lowertriangular_trisolv(n, L_baseline, x_baseline, b_baseline);

        // Compute baseline solution
        trisolv_baseline(n, L_baseline, x_baseline, b_baseline);
    }

    // Compute trisolv using MPI
    trisolv_mpi_v0(n, L, x, b); // This function needs to be implemented

    if (rank == 0) {
        // Check results in process 0
        for (int i = 0; i < n * n; i++) {
            EXPECT_NEAR(L[i], L_baseline[i], 1e-6);
        }
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
            EXPECT_NEAR(b[i], b_baseline[i], 1e-6);
        }

        // Free the allocated memory
        MPI_Free_mem(L);
        MPI_Free_mem(x);
        MPI_Free_mem(b);
        MPI_Free_mem(L_baseline);
        MPI_Free_mem(x_baseline);
        MPI_Free_mem(b_baseline);
    }
}

TEST(trisolvMPITest, DifferentSizes) {
    std::vector<int> n_vec = {0, 1, 10, 100}; // Example size
    for (auto n : n_vec){
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        double *L, *x, *b;
        double *L_baseline, *x_baseline, *b_baseline;

        // Allocate memory using MPI
        MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b);

        MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L_baseline);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b_baseline);

        if (rank == 0) {
            // Initialize matrices and vectors on process 0
            init_trisolv(n, L, x, b);
            init_trisolv(n, L_baseline, x_baseline, b_baseline);

            // Compute baseline solution
            trisolv_baseline(n, L_baseline, x_baseline, b_baseline);
        }

        // Compute trisolv using MPI
        trisolv_mpi_v0(n, L, x, b); // This function needs to be implemented

        if (rank == 0) {
            // Check results in process 0
            for (int i = 0; i < n * n; i++) {
                EXPECT_NEAR(L[i], L_baseline[i], 1e-6);
            }
            for (int i = 0; i < n; i++) {
                EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
                EXPECT_NEAR(b[i], b_baseline[i], 1e-6);
            }

            // Free the allocated memory
            MPI_Free_mem(L);
            MPI_Free_mem(x);
            MPI_Free_mem(b);
            MPI_Free_mem(L_baseline);
            MPI_Free_mem(x_baseline);
            MPI_Free_mem(b_baseline);
        }
    }

}
// Additional tests for RandomInitialization and DifferentSizes follow similar patterns

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}