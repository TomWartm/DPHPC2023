//
// Created by gao on 27.11.23.
//

#ifndef DPHPC2023_TIRSOLV_MPI_GAO_H
#define DPHPC2023_TIRSOLV_MPI_GAO_H

double trisolv_mpi_gao(int size, int rank, int NDEF, double*& A, double*& x, double*& b, void (*)(int, double*, double*, double*));

#endif //DPHPC2023_TIRSOLV_MPI_GAO_H