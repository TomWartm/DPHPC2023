#include "helpers/measure.h"
#include <iostream>
#include "gemver/gemver_baseline.h"
#include "gemver/openmp/gemver_openmp.h"
#include "omp.h"


int main(int argc, char *argv[])
{
    // open file
    int num_threads;    
    #pragma omp parallel
    {
      num_threads = omp_get_num_threads();
    }
    std::string filePath = "./results/gemver/output_gemver_openmp_T" + std::to_string(num_threads) + ".csv";
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
            measure_gemver((std::string) "baseline", &kernel_gemver, n, outputFile);
            measure_gemver((std::string) "openblas", &kernel_gemver_openblas, n, outputFile);
            measure_gemver((std::string) "openmp", &gemver_openmp_v4, n, outputFile);
        }
    }
    outputFile.close();

    return 0;
}
