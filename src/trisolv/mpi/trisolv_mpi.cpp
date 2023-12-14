#include "trisolv_mpi.h"
#include <mpi.h>
#include <stdlib.h>
#include <cassert>
#include <cmath>
#include <omp.h>

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
void trisolv_mpi_isend(int n, double* L, double* x, double* b)
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
* Every time a process finishes its part, a new group that exludes
*   that process is created
*/
void trisolv_mpi_onesided(int n, double* L, double* x, double* b)
{
    const int block_size = 8; //per cache line: 8 x 8 bytes = 64 bytes

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
        delete[] origin_ranks;
    }

    assert(n % (world_size * block_size) == 0);
    int process_size = n / world_size;
    int process_start = world_rank * process_size;
    int process_end = (world_rank+1) * process_size;
    
    //only initialize the necessary part
    for (i = process_start; i < process_end; i++) {
        x[i] = b[i];
    }

    //OPEN WINDOW
    MPI_Win_create(x, n * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x_win);

    //GET PHASE
    for (i = 0; i < process_start; i += block_size) {
        //create group containing the target process
        if (i % process_size == 0) {
            target_rank++;
            MPI_Group_incl(group_world, 1, &target_rank, &group_target);
        }

        MPI_Win_start(group_target, 0, x_win);
            double x_block[block_size];
            MPI_Get(x_block, block_size, MPI_DOUBLE, target_rank, i, block_size, MPI_DOUBLE, x_win);
        MPI_Win_complete(x_win);

        for (int k = process_start; k < process_end; k++) {
            double temp = 0;
            //could be unrolled for fixed block_size
            for (int line_idx = 0, j = i; line_idx < block_size; line_idx++, j++) { 
                temp += L[k*n + j] * x_block[line_idx];
            }
            x[k] -= temp;
        }
    }

    //COMPUTE PHASE
    if (world_rank != world_size-1) { //not the last process
        for (i = process_start; i < process_end; i++) {
            //compute
            double temp = 0;
            for (j = process_start; j < i; j++) {
                temp += L[i*n + j] * x[j];
            }
            x[i] = (x[i] - temp) / L[i*n + i];
            //give access to other processes
            if ((i + 1) % block_size == 0) {
                MPI_Win_post(group_origin, 0, x_win);
                MPI_Win_wait(x_win);
            }
        }
    }
    else { //last process doesn't do synchronization
        for (i = process_start; i < process_end; i++) {
            //compute
            double temp = 0;
            for (j = process_start; j < i; j++) {
                temp += L[i*n + j] * x[j];
            }
            x[i] = (x[i] - temp) / L[i*n + i];
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


void trisolv_mpi_gao(int n, double* A, double* x, double* b) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int block_size = 8;

    int rows, A_size;
    int std_rows = std::ceil((double)n / size);
	int n_mod_std_rows = n % std_rows;
    if (rank == 0) {
        A_size = n;
        rows = std_rows;
    }
    else if (rank == size - 1 && n_mod_std_rows != 0) {
        A_size = std_rows;
        rows = n % std_rows;
    }
    else {
        A_size = std_rows;
        rows = std_rows;
    }

    int sender, remain_rows, send_count, j_plus_k, A_pos1, A_pos2;
    double x_tmp;
    int rank_x_std_rows = rank * std_rows;

    for (int j = 0; j < n; j += send_count) {
        sender = j / std_rows;
        if (sender == size - 1 && n_mod_std_rows != 0) remain_rows = n_mod_std_rows;
        else remain_rows = std_rows;
        remain_rows -= j % std_rows;
        send_count = remain_rows >= block_size || remain_rows == 0 ? block_size: remain_rows;
        
        if (rank == sender) { //Compute small triangular matrix of size BLOCK_SIZE
        	A_pos1 = j - rank_x_std_rows + j * A_size;
        	for (int k = 0; k < send_count; ++k) {
        		j_plus_k = j + k;
        		x[j_plus_k] = b[j_plus_k] / A[A_pos1];
        		x_tmp = x[j_plus_k];
        		A_pos2 = A_pos1 + 1 - k;
        		for (int l = 1; l < send_count; ++l) {
        			b[j + l] -= A[A_pos2] * x_tmp;
        			A_pos2 += 1;
        		}
        		A_pos1 += 1 + A_size;
        	}
        }
        
		MPI_Bcast(x + j, send_count, MPI_DOUBLE, sender, MPI_COMM_WORLD);
		
		A_pos1 = j * A_size;
        for (int k = 0; k < send_count; ++k) { //b - A * x
        	x_tmp = x[j + k];
	        for (int i = 0; i < rows; ++i) {  		        	
    	        b[rank_x_std_rows + i] -= A[A_pos1 + i] * x_tmp;
    	    }
    	    A_pos1 += A_size;
    	}
    }
}


/*
* Use openMP to speedup the sequential parts
*/
void trisolv_mpi_onesided_openmp(int n, double* L, double* x, double* b)
{
    const int block_size = 8; //per cache line: 8 x 8 bytes = 64 bytes

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
        delete[] origin_ranks;
    }

    assert(n % (world_size * block_size) == 0);
    int process_size = n / world_size;
    int process_start = world_rank * process_size;
    int process_end = (world_rank+1) * process_size;
    
    //only initialize the necessary part
    #pragma omp parallel for
        for (i = process_start; i < process_end; i++) {
            x[i] = b[i];
        }

    //OPEN WINDOW
    MPI_Win_create(x, n * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &x_win);

    //GET PHASE
    for (i = 0; i < process_start; i += block_size) {
        //create group containing the target process
        if (i % process_size == 0) {
            target_rank++;
            MPI_Group_incl(group_world, 1, &target_rank, &group_target);
        }

        MPI_Win_start(group_target, 0, x_win);
            double x_block[block_size];
            MPI_Get(x_block, block_size, MPI_DOUBLE, target_rank, i, block_size, MPI_DOUBLE, x_win);
        MPI_Win_complete(x_win);

        #pragma omp parallel for
        for (int k = process_start; k < process_end; k++) {
            double temp = 0;
            //could be unrolled for fixed block_size
            for (int line_idx = 0, j = i; line_idx < block_size; line_idx++, j++) { 
                temp += L[k*n + j] * x_block[line_idx];
            }
            x[k] -= temp;
        }
    }

    //COMPUTE PHASE
    if (world_rank != world_size-1) { //not the last process
        for (i = process_start; i < process_end; i++) {
            //compute
            double temp = 0;
            for (j = process_start; j < i; j++) {
                temp += L[i*n + j] * x[j];
            }
            x[i] = (x[i] - temp) / L[i*n + i];
            //give access to other processes
            if ((i + 1) % block_size == 0) { 
                MPI_Win_post(group_origin, 0, x_win);
                MPI_Win_wait(x_win);
            }
        }
    }
    else { //last process doesn't do synchronization
        for (i = process_start; i < process_end; i++) {
            //compute
            double temp = 0;
            for (j = process_start; j < i; j++) {
                temp += L[i*n + j] * x[j];
            }
            x[i] = (x[i] - temp) / L[i*n + i];
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
