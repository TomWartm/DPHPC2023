#include "helpers/measure.h"
#include <iostream>
#include "gemver/gemver_baseline.h"
#include "gemver/openmp/gemver_openmp.h"


int main(int argc, char *argv[])
{
    // open file
    std::string filePath = "./results/gemver/output_gemver_openmp.csv";
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1; // Return an error code
    }
    outputFile << "N;time [s];method" << std::endl;

    // run experiments
    int num_runs = 10;
    for (int n = 1000; n <= 7000; n +=1000)
    {
        for (int num_run = 0; num_run < num_runs; ++num_run)
        {
            // give user feedback
            std::cout << "N = " << n << std::endl;

            /////////////////////////// method 1 /////////////////////////////////////
            measure_gemver((std::string) "baseline", &kernel_gemver, n, outputFile);
            measure_gemver((std::string) "baseline blocked 1", &gemver_baseline_blocked_1, n, outputFile);
            //measure_gemver((std::string) "baseline blocked 2", &gemver_baseline_blocked_2, n, outputFile);
            measure_gemver((std::string) "openmp", &gemver_openmp_v1, n, outputFile);
            //measure_gemver((std::string) "openmp with padding", &gemver_openmp_v2, n, outputFile);
        }
    }
    outputFile.close();

    return 0;
}