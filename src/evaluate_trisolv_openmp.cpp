#include "helpers/measure.h"
#include "omp.h"
#include <iostream>
#include "trisolv/trisolv_baseline.h"
#include "trisolv/openmp/trisolv_openmp.h"

int main(int argc, char *argv[])
{
    // open file
    std::string filePath = "./results/trisolv/output_trisolv_openmp.csv";
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1; // Return an error code
    }
    outputFile << "N;time [s];method" << std::endl;

    // run experiments
    int num_runs = 20;
    omp_set_num_threads(NUM_THREADS);
    for (int n = 4000; n <= 40000; n += 4000)
    {
        for (int num_run = 0; num_run < num_runs; ++num_run)
        {
            std::cout << "N = " << n << std::endl;
            measure_trisolv((std::string) "baseline", &trisolv_baseline, n, outputFile);
            measure_trisolv((std::string) "openblas", &trisolv_openblas, n, outputFile);
            measure_trisolv((std::string) "openmp", &trisolv_openmp, n, outputFile);
            measure_trisolv((std::string) "openmp 2", &trisolv_openmp_2, n, outputFile);
            measure_trisolv((std::string) "openmp lowspace", &trisolv_openmp_lowspace, n, outputFile);
        }
    }
    outputFile.close();

    return 0;
}