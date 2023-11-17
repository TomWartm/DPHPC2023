#include "helpers/measure.h"
#include <iostream>
#include <fstream>
#include "gemver/gemver_baseline.h"

int main(int argc, char *argv[])
{
    // open file
    std::string filePath = "./results/gemver/output_gemver.csv";
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1; // Return an error code
    }
    outputFile << "N;time [s];method" << std::endl;

    // run experiments
    int num_runs = 5;
    for (int n = 4; n <= 10000; n *= 2)
    {
        for (int num_run = 0; num_run < num_runs; ++num_run)
        {
            // give user feedback
            std::cout << "N = " << n << std::endl;

            /////////////////////////// method 1 /////////////////////////////////////

            measure_gemver((std::string) "baseline", &kernel_gemver, n, outputFile);
        }
    }
    outputFile.close();

    return 0;
}