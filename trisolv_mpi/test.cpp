#include <iostream>
#include <omp.h>

int main() {
	omp_set_num_threads(16);
	std::cout << omp_get_max_threads() << "\n";
	return 0;
}
