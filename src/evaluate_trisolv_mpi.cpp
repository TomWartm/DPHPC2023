#include "helpers/mpi/measure.h"
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cmath>
#include "trisolv/mpi/trisolv_mpi.h"
#include "trisolv/mpi/trisolv_mpi_gao.h"

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
    int num_runs = 1;
    int n_max = std::pow(2, POW);
    for (int n = 128; n <= n_max; n *= 2)
    {

        for (int num_run = 0; num_run < num_runs; ++num_run)
        {

            // give user feedback
            //std::cout << "N = " << n << std::endl;

            /////////////////////////// baseline /////////////////////////////////////

            //measure_trisolv_mpi((std::string) "trisolv_mpi", &kernel_trisolv_mpi, n, outputFile);
            measure_trisolv_baseline(n, outputFile);

            /////////////////////////// method gao ///////////////////////////////////
//            measure_trisolv_mpi(n, outputFile);
//            measure_trisolv_naive(n, outputFile);
//            measure_trisolv_mpi_double(n, outputFile);
//            measure_trisolv_mpi_single(n, outputFile);
            measure_trisolv_mpi(n, outputFile, trisolv_mpi_gao_any, "4");
        }
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::cout << rank << "DONE\n";
    outputFile.close();
	std::cout << rank << "DONE2\n";
    MPI_Finalize();
   	std::cout << rank << "DONE3\n";
    return 0;
}
