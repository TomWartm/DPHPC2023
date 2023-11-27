#include "helpers/mpi/measure.h"
#include <iostream>
#include <fstream>
#include <mpi.h>
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
    int num_runs = 5;
    for (int n = 4000; n <= 8000; n *= 2)
    {

        for (int num_run = 0; num_run < num_runs; ++num_run)
        {

            // give user feedback
            std::cout << "N = " << n << std::endl;

            /////////////////////////// method 1 /////////////////////////////////////

            //measure_trisolv_mpi((std::string) "trisolv_mpi", &kernel_trisolv_mpi, n, outputFile);

            /////////////////////////// method gao ///////////////////////////////////

            measure_trisolv_mpi(n, outputFile);
        }
    }

    outputFile.close();

    MPI_Finalize();
    return 0;
}