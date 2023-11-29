#include <mpi.h>
#include <stdlib.h>
#include <cassert>
#include "trisolv_mpi.h"

void trisolv_mpi_v0(int n, double* L, double* x, double* b){
    for (int i = 0; i < n; i++)
    {
        x[i] = b[i];
        for (int j = 0; j <i; j++)
        {
            double tmp = -L[i * n + j] * x[j];
            x[i] = x[i] +tmp;
        }
        x[i] = x[i] / L[i * n + i];
    }
}

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
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //for simplicity
    assert(n % world_size == 0);
    int process_size = n / world_size;
    int process_start = world_rank * process_size;
    int process_end = (world_rank+1) * process_size;

    // //send b and L and initialize x
    // if (world_rank == 0) {
    //     MPI_Request *send_req_b = (MPI_Request *)malloc((world_size-1) * sizeof(MPI_Request));
    //     MPI_Request *send_req_L = (MPI_Request *)malloc((world_size-1) * sizeof(MPI_Request));
    //     for (int p = 1; p < world_size; p++) {
    //         MPI_Isend(&b[p*process_size], process_size, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &send_req_b[p-1]);
    //         MPI_Isend(&L[n*p*process_size], process_size * n, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &send_req_L[p-1]);
    //     }
    //     MPI_Waitall(world_size-1, send_req_b, MPI_STATUSES_IGNORE);
    //     MPI_Waitall(world_size-1, send_req_L, MPI_STATUSES_IGNORE);
    //     free(send_req_b);
    //     free(send_req_L);
    //     for (i = 0; i < process_size; i++) {
    //         x[i] = b[i];
    //     }
    // }
    // else { //receive b and L and initialize x
    //     MPI_Recv(&b[process_start], process_size, MPI_DOUBLE, 0, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     MPI_Recv(&L[n*process_start], n * process_size, MPI_DOUBLE, 0, world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     for (i = process_start; i < process_end; i++) {
    //         x[i] = b[i];
    //     }
    // }
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
    if (world_rank < world_size-1) { //not last process
        int nof_send_to = world_size - world_rank - 1;
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
            for (int send_to = world_rank + 1; send_to < world_size; send_to++) {
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
    if (world_rank == 0) {
        //receive them in order
        for (int p = 1; p < world_size; p++) {
            MPI_Recv(&x[p * process_size], process_size, MPI_DOUBLE, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else {
        MPI_Send(&x[process_start], process_size, MPI_DOUBLE, 0, world_rank, MPI_COMM_WORLD);
    }
}


/*
* Same approach, but with remote memory accesses
* Every time a process finishes its part, a new communicator that exludes
*   that process is created
*/
void kernel_trisolv_mpi_onesided(int n, double* L, double* x, double* b)
{
    const int doubles_per_line = 8; //per cache line: 8 x 8 bytes = 64 bytes

    int i, j;

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Win x_win;

    MPI_Group group_world;
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);

    MPI_Group group_target;
    int target_rank = -1;

    MPI_Group group_origin;
    if (world_rank != world_size - 1) {
        int num_origins = world_size - world_rank - 1;
        int *origin_ranks = new int[num_origins];
        for (i = 0, j = world_rank + 1; i < num_origins; i++, j++) {
            origin_ranks[i] = j;
        }
        MPI_Group_incl(group_world, num_origins, origin_ranks, &group_origin);
    }

    assert(n % (world_size * doubles_per_line) == 0);
    int process_size = n / world_size;
    int process_start = world_rank * process_size;
    int process_end = (world_rank+1) * process_size;

    // //get relevant parts of b and L from process 0
    // MPI_Win b_win, L_win;
    // MPI_Win_create(b, n * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &b_win);
    // MPI_Win_create(L, n * n * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &L_win);
    // if (world_rank != 0) {
    //     MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, b_win);
    //         MPI_Get(&b[process_start], process_size, MPI_DOUBLE, 0, process_start, process_size, MPI_DOUBLE, b_win);
    //     MPI_Win_unlock(0, b_win);
    //     MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, L_win);
    //         MPI_Get(&L[n*process_start], n*process_size, MPI_DOUBLE, 0, n*process_start, n*process_size, MPI_DOUBLE, L_win);
    //     MPI_Win_unlock(0, L_win);
    // }
    // MPI_Win_free(&b_win);
    // MPI_Win_free(&L_win);
    
    //only initialize the necessary part
    for (i = process_start; i < process_end; i++) {
        x[i] = b[i];
    }

    //OPEN WINDOW
    MPI_Win_create(x, n * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x_win);

    //GET PHASE
    for (i = 0; i < process_start; i += doubles_per_line) {
        //create group containing the target process
        if (i % process_size == 0) {
            target_rank++;
            MPI_Group_incl(group_world, 1, &target_rank, &group_target);
        }

        MPI_Win_start(group_target, 0 /*MPI_MODE_NOCHECK? */, x_win);
            double x_line[doubles_per_line];
            MPI_Get(x_line, doubles_per_line, MPI_DOUBLE, target_rank, i, doubles_per_line, MPI_DOUBLE, x_win);
        MPI_Win_complete(x_win);

        for (int k = process_start; k < process_end; k++) {
            j = i;
            //this loop can be unrolled for fixed doubles_per_line
            for (int line_idx = 0, j = i; line_idx < doubles_per_line; line_idx++, j++) { 
                x[k] -= L[k*n + j] * x_line[line_idx];
            }
        }
    }

    //COMPUTE PHASE
    if (world_rank != world_size-1) { //not the last process
        for (i = process_start; i < process_end; i++) {
            //compute
            for (j = process_start; j < i; j++) {
                x[i] -= L[i*n + j] * x[j];
            }
            x[i] /= L[i*n + i];
            if ((i + 1) % doubles_per_line == 0) { //give access to other processes
                MPI_Win_post(group_origin, 0 /* MPI_MODE_NOCHECK? */, x_win);
                MPI_Win_wait(x_win);
            }
        }
    }
    else { //last process doesn't do synchronization
        for (i = process_start; i < process_end; i++) {
            //compute
            for (j = process_start; j < i; j++) {
                x[i] -= L[i*n + j] * x[j];
            }
            x[i] /= L[i*n + i];
        }
    }

    if (world_rank != 0) {
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, x_win);
        MPI_Put(&x[process_start], process_size, MPI_DOUBLE, 0, process_start, process_size, MPI_DOUBLE, x_win);
        MPI_Win_unlock(0, x_win);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&x_win);
}