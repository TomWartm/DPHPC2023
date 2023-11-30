#pragma once

void rowMaj_to_colMaj(int N, double* source, double* target) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            target[j * N + i] = source[i * N + j];
        }
    }
}

void get_partial(int size, int target_rank, int N, double* source, double* target) {
	int std_rows = std::ceil(1.0 * N / size);
	int starting_row = target_rank * std_rows;
    
    for (int j = 0; j < N; ++j) {
    	for (int i = 0; i < std_rows; ++i) {
    		if (i + starting_row >= N) target[j * std_rows + i] = -1;
    		else target[j * std_rows + i] = source[j * N + i + starting_row];
    	}
    }
}

void init_colMaj(int size, int rank, int N, double*& A, double*& x, double*& b, void (*init)(int, double*, double*, double*)) {
	int std_rows = std::ceil(1.0 * N / size);
    
	if (rank == 0) {
        double* tmp = new double[N * N];
        A = new double[N * N];
        x = new double[N];
        b = new double[N];
        init(N, tmp, x, b);
        rowMaj_to_colMaj(N, tmp, A);
        delete[] tmp;

       	double* partial_A = new double[std_rows * N];
        for (int n = 1; n < size; ++n) {
			get_partial(size, n, N, A, partial_A);
			MPI_Send(partial_A, N * std_rows, MPI_DOUBLE, n, 1, MPI_COMM_WORLD);
        }
        delete[] partial_A;
   		MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   		MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else {
        A = new double[std_rows * N];
        x = new double[N];
        b = new double[N];
        MPI_Recv(A, N * std_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   		MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   		MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}
