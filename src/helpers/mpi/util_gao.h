#pragma once

void rowMaj_to_colMaj(int N, double* source, double* target);

void get_partial(int size, int target_rank, int N, double* source, double* target);

void init_colMaj(int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*));
