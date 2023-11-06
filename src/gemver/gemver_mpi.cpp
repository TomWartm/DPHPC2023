#include <mpi.h>
void gemver_mpi_1(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{   
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("I am %d of %d\n", rank, size);
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = A[i * n + j] + u1[i] * v1[j] + u2[i] * v2[j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            x[i] = x[i] + beta * A[j * n + i] * y[j];

    for (int i = 0; i < n; i++)
        x[i] = x[i] + z[i];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            w[i] = w[i] + alpha * A[i * n + j] * x[j];

}
