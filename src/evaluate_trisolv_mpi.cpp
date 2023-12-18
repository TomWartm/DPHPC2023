#include "helpers/mpi/measure.h"
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "trisolv/mpi/trisolv_mpi.h"

int main(int argc, char *argv[])
{   
    MPI_Init(&argc, &argv);
    
    // open file
    std::string filePath = "./results/trisolv/output_trisolv_mpi.csv";
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1; // Return an error code
    }
    outputFile << "N;time [s];method" << std::endl;

	// run experiments
    int num_runs = NUM_RUNS;
    int n_min = std::pow(2, N_MIN);
    int n_max = std::pow(2, N_MAX);
    for (int n = n_min; n <= n_max; n *= 2)
    {
       for (int num_run = 0; num_run < num_runs; ++num_run)
        {

            // give user feedback
            //std::cout << "N = " << n << std::endl;

            //////////////////////// Baseline /////////////////////////////////////////////
            measure_trisolv_mpi((std::string) "trisolv_baseline", &trisolv_mpi_v0, n, outputFile);
            
            //////////////////////// BLAS /////////////////////////////////////////////////
            measure_trisolv_mpi((std::string) "trisolv_blas", &trisolv_blas, n, outputFile);            

            //////////////////////// With non blocking send ///////////////////////////////
            measure_trisolv_mpi((std::string) "trisolv_mpi_isend", &trisolv_mpi_isend, n, outputFile);  //isend and onesided don't work for matrix size > 8192,
	   												//comment them out for benchmarks exceeding that size

            /////////////////////// with rma //////////////////////////////////////////////
            measure_trisolv_mpi((std::string) "trisolv_mpi_onesided", &trisolv_mpi_onesided, n, outputFile);

            /////////////////////// method gao ////////////////////////////////////////////
            measure_trisolv_mpi((std::string) "trisolv_mpi_gao", &trisolv_mpi_gao, n, outputFile);

            measure_trisolv_mpi((std::string) "trisolv_mpi_onesided_openmp", &trisolv_mpi_onesided_openmp, n, outputFile);
        }   
    }

    outputFile.close();
    MPI_Finalize();
    return 0;
}
