#include <mpi.h>
#include <stdlib.h>
#include "trisolv_mpi.h"

/*
* Process 0 computes its part. Once a value of x is computed, it sends it to all processes with higher rank.
*   processes that receive a value start using it (in the "subtruction" part). Once process 0 has finished,
*   process 1 does the same with its part, and so on.
* It's way slower than the original, but hopefully it has potential...
* in both the receive and send phase openMP could be used
*/
void kernel_trisolv_mpi(int n, double* L, double* x, double* b)
{

    int i, j;
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    //for simplicity
    int process_size = n / mpi_size;
    int process_start = mpi_rank * process_size;
    int process_end = (mpi_rank+1) * process_size;

    //only initialize the necessary part
    for (i = process_start; i < process_end; i++) {
        x[i] = b[i];
    }

    //RECEIVE PHASE
    int receive_from = -1;
    for (i = 0; i < process_start; i++) {
        if (i % process_size == 0) { //switch to receiving from the next process
            receive_from++;
        }
        double xi; //x[i]
        MPI_Recv(&xi, 1, MPI_DOUBLE, receive_from, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int k = process_start; k < process_end; k++) {
            x[k] -= L[k*n + i] * xi;
        }
    }

    //SEND PHASE
    if (mpi_rank < mpi_size-1) { //not last process
        int nof_send_to = mpi_size - mpi_rank - 1;
        int req_idx = 0;
        //should change to "new" to keep it all c++
        MPI_Request *send_req = (MPI_Request *)malloc(process_size * nof_send_to * sizeof(MPI_Request));
        for (/*i already set*/; i < process_end; i++) {
            //compute
            for (j = process_start; j < i; j++) {
                x[i] -= L[i*n + j] * x[j];
            }
            x[i] /= L[i*n + i];
            //send
            for (int send_to = mpi_rank + 1; send_to < mpi_size; send_to++) {
                MPI_Isend(&x[i], 1, MPI_DOUBLE, send_to, i, MPI_COMM_WORLD, &send_req[req_idx]);
                req_idx++;
            }
        }
        MPI_Waitall(nof_send_to * process_size, send_req, MPI_STATUSES_IGNORE);
        free(send_req);
    }
    else {//last process doesn't send
        for (/*i already set*/; i < process_end; i++) {
            //compute
            for (j = process_start; j < i; j++) {
                x[i] -= L[i*n + j] * x[j];
            }
            x[i] /= L[i*n + i];
        }
    }

    //SEND BACK TO PROCESS 0
    //TODO
}