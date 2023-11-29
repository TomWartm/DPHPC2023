#ifndef MEASURE_H
#define MEASURE_H

#include <string>
#include <fstream>

void measure_gemver_mpi(std::string functionName,void (*func)(int, double, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*), int n, std::ofstream &outputFile);

void measure_trisolv_mpi(std::string functionName,void (*func)(int , double*, double*, double*), int n, std::ofstream &outputFile);

void measure_trisolv_naive(int n, std::ofstream &outputFile);

void measure_trisolv_mpi_single(int n, std::ofstream &outputFile);

void measure_trisolv_mpi_double(int n, std::ofstream &outputFile);

void measure_trisolv_mpi_combined(int n, std::ofstream &outputFile);

#endif
