#include <gtest/gtest.h>
#include <mpi.h>
#include "../../../src/trisolv/trisolv_baseline.h"
#include "../../../src/helpers/trisolv_init.h"
#include "../../../src/trisolv/mpi/trisolv_mpi.h"
#include "../../../src/helpers/mpi/util_gao.h"

/// @brief
/// @param
/// @param
TEST(trisolvMPITest, IdentityInitialization) {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 256; // Example size

    double *L, *b, *x, *x_baseline;

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);

    //////////////BASELINE
    if (rank == 0) {
        // Initialize matrices and vectors on process 0
        identity_trisolv(n, L, x_baseline, b);

        // Compute baseline solution
        trisolv_baseline(n, L, x_baseline, b);

    }

    //////////////METHOD TO TEST
    init_colMaj(n, L, x, b, &identity_trisolv);
    // Compute trisolv using MPI
    trisolv_mpi_gao(n, L, x, b);
    if (rank == 0) {
        // Check results in process 0
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Free_mem(L);
    MPI_Free_mem(b);
    MPI_Free_mem(x);
    MPI_Free_mem(x_baseline);
}

TEST(trisolvMPITest, RandomInitialization) {
int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 256; // Example size

    double *L, *b, *x, *x_baseline;

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);

    ////////////BASELINE
    if (rank == 0) {
        // Initialize matrices and vectors on process 0
        //identity_trisolv(n, L, x, b);
        random_trisolv(n, L, x_baseline, b);

        // Compute baseline solution
        trisolv_baseline(n, L, x_baseline, b);

    }

    ///////////METHOD TO TEST
    init_colMaj(n, L, x, b, &random_trisolv);
    // Compute trisolv using MPI
    trisolv_mpi_gao(n, L, x, b);
    if (rank == 0) {
        // Check results in process 0
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
   
    MPI_Free_mem(L);
    MPI_Free_mem(b);
    MPI_Free_mem(x);
    MPI_Free_mem(x_baseline);
}

TEST(trisolvMPITest, LTriangularInitialization) {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 256; // Example size

    double *L, *b, *x, *x_baseline;

    MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);

    ////////////BASELINE
    if (rank == 0) {
        // Initialize matrices and vectors on process 0
        //identity_trisolv(n, L, x, b);
        lowertriangular_trisolv(n, L, x_baseline, b);

        // Compute baseline solution
        trisolv_baseline(n, L, x_baseline, b);

    }

    ///////////METHOD TO TEST
    init_colMaj(n, L, x, b, &lowertriangular_trisolv);
    // Compute trisolv using MPI
    trisolv_mpi_gao(n, L, x, b);
    if (rank == 0) {
        // Check results in process 0
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Free_mem(L);
    MPI_Free_mem(b);
    MPI_Free_mem(x);
    MPI_Free_mem(x_baseline);
}

TEST(trisolvMPITest, DifferentSizes) {
    std::vector<int> n_vec = {128, 512, 1024, 4096}; // Example size
    for (auto n : n_vec){
        int rank, num_procs;
    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    	
        double *L, *b, *x, *x_baseline;

        MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &L);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &b);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_baseline);
	
    	if (rank == 0) {
    	    // Initialize matrices and vectors on process 0
    	    init_trisolv(n, L, x_baseline, b);
	
    	    // Compute baseline solution
    	    trisolv_baseline(n, L, x_baseline, b);
	
    	}
	
    	///////////METHOD TO TEST
        init_colMaj(n, L, x, b, &init_trisolv);
        // Compute trisolv using MPI
        trisolv_mpi_gao(n, L, x, b);
    	if (rank == 0) {
    	    // Check results in process 0
    	    for (int i = 0; i < n; i++) {
    	        EXPECT_NEAR(x[i], x_baseline[i], 1e-6);
    	    }
    	}
	    MPI_Barrier(MPI_COMM_WORLD);
	    
        MPI_Free_mem(L);
        MPI_Free_mem(b);
        MPI_Free_mem(x);
        MPI_Free_mem(x_baseline);
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
