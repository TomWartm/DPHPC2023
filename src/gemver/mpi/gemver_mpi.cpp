#include <mpi.h>
void gemver_mpi_1(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z, double *A_result, double *x_result, double *w_result)
{   
    /*
    This is a dummy version. It broadcasts the data to all processes and it computes entire result in every process.
    */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    
    // Broadcast the data to all other processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(A, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(u2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(z, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0){
        // compute values
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
    

    
    // gather all results into _result variables
    MPI_Reduce(A, A_result, n * n, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // just take max of all results of all processes
    MPI_Reduce(x, x_result, n, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);     
    MPI_Reduce(w, w_result, n, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);     

}



