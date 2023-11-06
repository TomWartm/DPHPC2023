#include "helpers/measure.h"
#include <iostream>
#include <fstream>
#include "gemver/gemver_baseline.h"
#include "gemver/gemver_mpi.h"
#include <mpi.h>

int main(int argc, char *argv[])
{   
    MPI_Init(&argc, &argv);
    // open file
    std::string filePath = "./results/output.csv";
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1; // Return an error code
    }
    outputFile << "N;time [s];method" << std::endl;

    // run experiments
    int num_runs = 2;
    for (int n = 4000; n <= 8000; n *= 2)
    {

        for (int num_run = 0; num_run < num_runs; ++num_run)
        {

            // give user feedback
            std::cout << "N = " << n << std::endl;

            /////////////////////////// method 1 /////////////////////////////////////

            measure_gemver((std::string) "baseline", &kernel_gemver, n, outputFile);
            
            measure_gemver((std::string) "mpi_1", &gemver_mpi_1, n, outputFile);
            
        }
    }

    outputFile.close();

    MPI_Finalize();
    return 0;
}