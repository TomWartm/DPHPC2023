#include <cstdlib>
#include "omp.h"
#include <iostream>
#include "trisolv/trisolv_baseline.h"
#include "trisolv/openmp/trisolv_openmp.h"
#include "helpers/trisolv_init.h"

int main(int argc, char *argv[])
{
    char* fun = argv[1];
    int n = atoi(argv[2]);
    
    double *L = (double *)malloc((n * n) * sizeof(double));
    double *x = (double *)malloc((n) * sizeof(double));
    double *b = (double *)malloc((n) * sizeof(double));

    omp_set_num_threads(NUM_THREADS);
    if (fun == (std::string) "openmp_lowspace") {
        init_trisolv_lowspace(n, L, x, b);
        trisolv_openmp_lowspace(n, L, x, b);
    } else if (fun == (std::string) "openblas"){
        init_trisolv(n, L, x, b);
        trisolv_openblas(n, L, x, b);
    } else if (fun == (std::string) "openmp"){
        init_trisolv(n, L, x, b);
        trisolv_openmp(n, L, x, b);
    } else if (fun == (std::string) "openmp_2"){
        init_trisolv(n, L, x, b);
        trisolv_openmp_2(n, L, x, b);
    } else if (fun == (std::string) "openmp_3"){
        init_trisolv_colmaj(n, L, x, b);
        trisolv_openmp_3(n, L, x, b);
    } else {
        std::cout << "Unknown function " << fun << std::endl;
    }
    
    free((void*)L);
    free((void*)x);
    free((void*)b);

    return 0;
}