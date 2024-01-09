#include "helpers/mpi/measure.h"
#include <iostream>
#include "gemver/mpi/gemver_mpi.h"
#include "gemver/mpi/gemver_mpi_openmp.h"
#include "omp.h"
#include <mpi.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int threads = omp_get_max_threads();
    std::string filePath = "./results/gemver/output_gemver_mpi_" + std::to_string(threads) + "_omp_threads_" + std::to_string(world_size) + "_mpi_tasks.csv";
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1; // Return an error code
    }
    outputFile << "N;time [s];method" << std::endl;

    // run experiments
    int num_runs = 20;
    for (int n = 4000; n <= 40000; n += 4000)
    {
        for (int num_run = 0; num_run < num_runs; ++num_run)
        {
            std::cout << "N = " << n << std::endl;
            measure_gemver_mpi((std::string) "mpi", &gemver_mpi_3_new, n, outputFile);
            measure_gemver_mpi((std::string) "hybrid", &gemver_mpi_3_new_openmp, n, outputFile);
        }
    }

    outputFile.close();

    MPI_Finalize();
    return 0;
}