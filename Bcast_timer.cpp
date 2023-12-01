#include <iostream>
#include <mpi.h>
#include <chrono>
#include <iomanip>

int main(int argc, char *argv[])
{   
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
   	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    double* A;
    for (int i = 1; i < 1024; i *= 2) {
    	A = new double[i];
    	if (rank == 0) {
    		for (int n = 0; n < i; ++n) A[n] = n;
        	start = std::chrono::high_resolution_clock::now();
        }
    	for (int j = 0; j < REPEAT; ++j) {
			MPI_Bcast(A, i, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    	}
    	if (rank == 0) {
	       	end = std::chrono::high_resolution_clock::now();
	    	const std::chrono::duration<double> diff = end - start;
    	    std::cout << std::fixed << std::setprecision(9) << std::left;
    	    std::cout << size << "\t" << i << "\t" << diff.count() / REPEAT << "\n";
    	}
    	delete[] A;
    }
    
    MPI_Finalize();
    return 0;
}
