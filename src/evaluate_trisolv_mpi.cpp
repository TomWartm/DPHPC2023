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
    int num_runs = 20;
    for (int n = 4000; n <= 16000; n += 4000)
    {
        for (int num_run = 0; num_run < num_runs; ++num_run)
        {
            std::cout << "N = " << n << std::endl;
            measure_trisolv_mpi((std::string) "mpi", &trisolv_mpi_isend, n, outputFile);  //isend and onesided don't work for matrix size > 8192,
            measure_trisolv_mpi((std::string) "mpi openmp hybrid", &trisolv_mpi_isend_openmp, n, outputFile);
        }   
    }

    outputFile.close();
    MPI_Finalize();
    return 0;
}
