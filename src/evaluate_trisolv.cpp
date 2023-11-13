#include "helpers/measure.h"
#include <iostream>
#include <fstream>
#include "trisolv/trisolv_baseline.h"
#include "trisolv/openmp/trisolv_openmp.h"

int main(int argc, char *argv[])
{
    // open file
    std::string filePath = "./results/trisolv/output_trisolv.csv";
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1; // Return an error code
    }
    outputFile << "N;time [s];method" << std::endl;

    // run experiments
    int num_runs = 20;
    for (int n = 4000; n <= 8000; n *= 2)
    {
        for (int num_run = 0; num_run < num_runs; ++num_run)
        {
            // give user feedback
            std::cout << "N = " << n << std::endl;

            /////////////////////////// method 1 /////////////////////////////////////

            measure_trisolv((std::string) "baseline", &trisolv_baseline, n, outputFile);
            measure_trisolv((std::string) "openmp", &trisolv_openmp, n, outputFile);
        }
    }
    outputFile.close();

    return 0;
}