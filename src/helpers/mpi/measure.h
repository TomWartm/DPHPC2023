#ifndef MEASURE_H
#define MEASURE_H

#include <string>
#include <fstream>

void measure_gemver_mpi(std::string functionName,void (*func)(int, double, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*), int n, std::ofstream &outputFile);

void measure_trisolv_mpi(std::string functionName,void (*func)(int , double*, double*, double*), int n, std::ofstream &outputFile);
void measure_trisolv_baseline(int n, std::ofstream &outputFile);
void measure_trisolv_mpi(int n, std::ofstream &outputFile);


void measure_trisolv_naive(int n, std::ofstream &outputFile);
void measure_trisolv_mpi_single(int n, std::ofstream &outputFile);
double measure_trisolv_mpi(int n, std::ofstream &outputFile, double (*solver)(int, int, int, double*&, double*&, double*&, void (*)(int, double*, double*, double*), int), const std::string& name, int block_size);


#endif
