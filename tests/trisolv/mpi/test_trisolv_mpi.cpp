#include <gtest/gtest.h>
#include <mpi.h>
#include "../../../src/trisolv/trisolv_baseline.h"
#include "../../../src/helpers/trisolv_init.h"
#include "../../../src/trisolv/mpi/trisolv_mpi.h"
#include "../../../src/trisolv/mpi/trisolv_mpi_gao.h"

/// @brief
/// @param
/// @param
TEST(trisolvMPITest, IdentityInitialization) {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 10; // Example size

    double *L = nullptr, *x = nullptr, *b = nullptr;
    double *L_baseline, *x_baseline, *b_baseline;

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b_baseline);

    if (rank == 0) {
        // Initialize matrices and vectors on process 0
        //identity_trisolv(n, L, x, b);
        identity_trisolv(n, L_baseline, x_baseline, b_baseline);

        // Compute baseline solution
        trisolv_baseline(n, L_baseline, x_baseline, b_baseline);

    }

    MPI_Free_mem(L_baseline);
    MPI_Free_mem(b_baseline);

    // Compute trisolv using MPI
    trisolv_mpi_gao(num_procs, rank, n, L, x, b, identity_trisolv);
    if (rank == 0) {
        // Check results in process 0
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Free_mem(x_baseline);
    delete[] L;
    delete[] x;
    delete[] b;
}

TEST(trisolvMPITest, RandomInitialization) {
int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 10; // Example size

    double *L = nullptr, *x = nullptr, *b = nullptr;
    double *L_baseline, *x_baseline, *b_baseline;

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b_baseline);

    if (rank == 0) {
        // Initialize matrices and vectors on process 0
        //identity_trisolv(n, L, x, b);
        random_trisolv(n, L_baseline, x_baseline, b_baseline);

        // Compute baseline solution
        trisolv_baseline(n, L_baseline, x_baseline, b_baseline);

    }

    MPI_Free_mem(L_baseline);
    MPI_Free_mem(b_baseline);

    // Compute trisolv using MPI
    trisolv_mpi_gao(num_procs, rank, n, L, x, b, random_trisolv);
    if (rank == 0) {
        // Check results in process 0
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Free_mem(x_baseline);
    delete[] L;
    delete[] x;
    delete[] b;
}

TEST(trisolvMPITest, LTriangularInitialization) {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 10; // Example size

    double *L = nullptr, *x = nullptr, *b = nullptr;
    double *L_baseline, *x_baseline, *b_baseline;

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b_baseline);

    if (rank == 0) {
        // Initialize matrices and vectors on process 0
        //identity_trisolv(n, L, x, b);
        lowertriangular_trisolv(n, L_baseline, x_baseline, b_baseline);

        // Compute baseline solution
        trisolv_baseline(n, L_baseline, x_baseline, b_baseline);

    }

    MPI_Free_mem(L_baseline);
    MPI_Free_mem(b_baseline);

    // Compute trisolv using MPI
    trisolv_mpi_gao(num_procs, rank, n, L, x, b, lowertriangular_trisolv);
    if (rank == 0) {
        // Check results in process 0
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Free_mem(x_baseline);
    delete[] L;
    delete[] x;
    delete[] b;
}

TEST(trisolvMPITest, DifferentSizes) {
    std::vector<int> n_vec = {1, 100, 1000, 10000}; // Example size
    for (auto n : n_vec){
        int rank, num_procs;
    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    	double *L = nullptr, *x = nullptr, *b = nullptr;
    	double *L_baseline, *x_baseline, *b_baseline;
	
    	MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L_baseline);
    	MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
    	MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b_baseline);
	
    	if (rank == 0) {
    	    // Initialize matrices and vectors on process 0
    	    //identity_trisolv(n, L, x, b);
    	    lowertriangular_trisolv(n, L_baseline, x_baseline, b_baseline);
	
    	    // Compute baseline solution
    	    trisolv_baseline(n, L_baseline, x_baseline, b_baseline);
	
    	}
	
    	MPI_Free_mem(L_baseline);
    	MPI_Free_mem(b_baseline);
	
    	// Compute trisolv using MPI
        trisolv_mpi_gao(num_procs, rank, n, L, x, b, lowertriangular_trisolv);
    	if (rank == 0) {
    	    // Check results in process 0
    	    for (int i = 0; i < n; i++) {
    	        EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
    	    }
    	}
	    MPI_Barrier(MPI_COMM_WORLD);
	    MPI_Free_mem(x_baseline);
	    delete[] L;
	   	delete[] x;
	    delete[] b;
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
