#include "helpers/mpi/measure.h"
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "trisolv/mpi/trisolv_mpi.h"
#include "trisolv/mpi/trisolv_mpi_gao.h"

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
    int num_runs = 10;
    int n_min = std::pow(2, N_MIN);
    int n_max = std::pow(2, N_MAX);
    
    for (int n = n_min; n <= n_max; n *= 2)
    {

       *for (int num_run = 0; num_run < num_runs; ++num_run)
        {

            // give user feedback
            //std::cout << "N = " << n << std::endl;

            /////////////////////////// baseline /////////////////////////////////////

            //measure_trisolv_mpi((std::string) "trisolv_mpi", &kernel_trisolv_mpi, n, outputFile);
            measure_trisolv_baseline(n, outputFile);

            /////////////////////////// method gao ///////////////////////////////////
//            measure_trisolv_mpi(n, outputFile);
//            measure_trisolv_naive(n, outputFile);
            measure_trisolv_mpi(n, outputFile, trisolv_mpi_gao_any, "mono", 1);
//            measure_trisolv_mpi_single(n, outputFile);
//            measure_trisolv_mpi_double(n, outputFile);
//            measure_trisolv_mpi(n, outputFile, trisolv_mpi_gao_any, "quad", 4);
            measure_trisolv_mpi(n, outputFile, trisolv_mpi_gao_any, "octa", 8);

		
        }      
    }
    /*
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    std::vector<std::vector<std::vector<double>>> results;
    for (int N = n_min; N <= n_max; N *= 2) {
    	std::vector<double> block_medians;
	    for (int b = 1; b <= MAX_BLOCK; b *= 2) {
    		std::vector<double> N_results;
    		int num_runs = std::max(16384 / (N / b), 1);
    		for (int i = 0; i < num_runs; ++i) {
//    			if (rank == 0) std::cout << b << " " << N << " " << i << "\n";
    			N_results.push_back(measure_trisolv_mpi(N, outputFile, trisolv_mpi_gao_any, std::to_string(b), b));
    		}
    		std::sort(N_results.begin(), N_results.end());
	        double median = N_results.at(N_results.size() / 2);
    		if (rank == 0) {
	    		std::cout << std::fixed << std::setprecision(9) << std::left;
	    		std::cout << N << "\t" << b << "\t" << median << "\n";
	    	}
	    	block_medians.push_back(median);
    	}
    	auto it = std::min_element(block_medians.begin(), block_medians.end());
    	if (rank == 0) std::cout << "Best block size for " << N << " is " << int(std::pow(2, it - block_medians.begin())) << "\n\n";  	
//    	results.push_back(block_results);
    }*/

    outputFile.close();
    MPI_Finalize();
    return 0;
}
