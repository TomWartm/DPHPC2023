#ifndef MEASURE_H
#define MEASURE_H

#include <string>
#include <fstream>

void measure_gemver_mpi(std::string functionName,void (*func)(int, double, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*), int n, std::ofstream &outputFile);

#endif
