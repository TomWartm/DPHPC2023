#include "helpers/measure.h"
#include <iostream>
#include <fstream>
#include "gemver/gemver_baseline.h"


int main()
{
    // open file
    std::string filePath = "results/output.csv";
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1; // Return an error code
    }
    outputFile << "N;time [s];method" << std::endl;

    // run experiments
    int num_runs = 10;
    for (int num_run = 0; num_run < num_runs; ++num_run)
    {

        for (int n = 1; n <= 4000; n *= 2)
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