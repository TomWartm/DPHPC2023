#include "helpers/mpi/measure.h"
#include <iostream>
#include <fstream>
#include "gemver/mpi/gemver_mpi.h"
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    // open file
    std::string filePath = "./results/gemver/output_gemver_mpi.csv";
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1; // Return an error code
    }
    outputFile << "N;time [s];method" << std::endl;

    // run experiments
    int num_runs = 2;
    for (int n = 4; n <= 10000; n *= 2)
    {

        for (int num_run = 0; num_run < num_runs; ++num_run)
        {

            // give user feedback
            std::cout << "N = " << n << std::endl;

            /////////////////////////// method 1 /////////////////////////////////////

            measure_gemver_mpi((std::string) "mpi_baseline", &gemver_mpi_1, n, outputFile);
            measure_gemver_mpi((std::string) "gemver_mpi_2", &gemver_mpi_2, n, outputFile);
        }
    }

    outputFile.close();

    MPI_Finalize();
    return 0;
}