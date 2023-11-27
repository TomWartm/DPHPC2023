//
// Created by gao on 27.11.23.
//

#ifndef DPHPC2023_TIRSOLV_MPI_GAO_H
#define DPHPC2023_TIRSOLV_MPI_GAO_H

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>

template <class Init>
double trisolv_mpi_gap(int size, int rank, int NDEF, double* A, double* x, double* b, Init init);

#endif //DPHPC2023_TIRSOLV_MPI_GAO_H
