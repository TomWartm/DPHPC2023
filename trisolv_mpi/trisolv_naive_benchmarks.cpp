#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <iomanip>
#include <cmath>

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trisolv(int n,
		    double* A,
		    double* x,
		    double* b)
{
  int i, j;

#pragma scop
	for (i = 0; i < n; i++)
    {
		x[i] = b[i];
		for (j = 0; j < i; j++)
			x[i] -= A[i * n + j] * x[j];
		x[i] = x[i] / A[i * n + i];
    }
#pragma endscop

}

void init(int N, double* A, double* x, double* b) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			A[i * N + j] = 1.0 * (j <= i);
		}
	}
	int k = 0;
	for (int i = 0; i < N; ++i) {
		k += i + 1;
		x[i] = 0.0;
		b[i] = k;
	}
}



int main() {
	for (int N = 6; N <= POW; ++N) {
		int n = std::pow(2, N);
		double* A = (double*)malloc(n * n * sizeof(double));
		double* x = (double*)malloc(n * sizeof(double));
		double* b = (double*)malloc(n * sizeof(double));
		init(n, A, x, b);
		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		start = std::chrono::high_resolution_clock::now();
		
		kernel_trisolv(n, A, x, b);
		
		end = std::chrono::high_resolution_clock::now();
		const std::chrono::duration<double> diff = end - start;
		std::cout << n << " ";
		std::cout << std::fixed << std::setprecision(9) << std::left;
		std::cout << diff.count() << '\n';
	}
	return 0;
}