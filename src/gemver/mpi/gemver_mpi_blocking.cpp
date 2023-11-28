#include <mpi.h>
#include <iomanip>
#include "../gemver_baseline.h"

void gemver_mpi_4(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z, double *A_result, double *x_result, double *w_result)
{
    /*
    Same as gemver_mpi_3 but with blocking of 4
    */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // assign memory to variables that are used on all processes
    if (rank!=0){
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v1);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v2);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &y);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);

        MPI_Alloc_mem(n * n * sizeof(double), MPI_INFO_NULL, &A_result);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x_result);
        MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &w_result);
    }

    /*
    Part 1: A_result = A + u1*v1 + u2*v2
    */

    // split u1, u2 and A into blocks and distribute over all processes
    int blockSize = n / size;
    int remSize = n % size;
    // send vectors u1, u2: all processes get size: blockSize, except last process which gets the remaining elements
    int *displsVector = (int *)malloc(size * sizeof(int));  // stores start index of each block
    int *scountsVector = (int *)malloc(size * sizeof(int)); // stores number of elements in each block
    for (int i = 0; i < size; i++)
    {
        if (i < size - 1)
        {
            scountsVector[i] = blockSize;
        }
        else
        {
            scountsVector[i] = blockSize + remSize;
        }
        displsVector[i] = i * blockSize;
    }
    int localSize = scountsVector[rank]; // for al processes < size -1 == blockSize else blockSize + remSize
    double *local_u1 = (double *)malloc(localSize * sizeof(double));
    double *local_u2 = (double *)malloc(localSize * sizeof(double));

    MPI_Request send_req_u1, send_req_u2, send_req_A;
    MPI_Request recv_req_A_result;

    MPI_Iscatterv(u1, scountsVector, displsVector, MPI_DOUBLE, local_u1, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_req_u1);
    MPI_Iscatterv(u2, scountsVector, displsVector, MPI_DOUBLE, local_u2, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_req_u2);

    // send array A: all processes get blockSize rows, except last process which gets the remaining elements
    int *displsMatrix = (int *)malloc(size * sizeof(int));  // stores start index of each block
    int *scountsMatrix = (int *)malloc(size * sizeof(int)); // stores number of elements in each block
    for (int i = 0; i < size; i++)
    {
        if (i < size - 1)
        {
            scountsMatrix[i] = blockSize * n;
        }
        else
        {
            scountsMatrix[i] = (blockSize + remSize) * n;
        }
        displsMatrix[i] = i * blockSize * n;
    }

    double *local_A = (double *)malloc(localSize * n * sizeof(double));
    MPI_Iscatterv(A, scountsMatrix, displsMatrix, MPI_DOUBLE, local_A, localSize * n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_req_A);

    // send v1, v2 entirely to all processes
    MPI_Bcast(v1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);




    // Wait for the non-blocking communication to complete
    MPI_Wait(&send_req_u1, MPI_STATUS_IGNORE);
    MPI_Wait(&send_req_u2, MPI_STATUS_IGNORE);
    MPI_Wait(&send_req_A, MPI_STATUS_IGNORE);

    // compute  A += u1 * v1 + u2 * v2

    part_1(localSize, n, local_A, local_u1, local_u2, v1, v2);


    // get local results back to process 0
    MPI_Igatherv(local_A, localSize * n, MPI_DOUBLE, A_result, scountsMatrix, displsMatrix, MPI_DOUBLE, 0, MPI_COMM_WORLD, &recv_req_A_result);
    
    /*
    Part 2: x_result = x + beta * (A_result)T * y + z
    */

    // array (A_result)T: all processes get blockSize columns, except last process which gets the remaining elements

    // https://stackoverflow.com/questions/9269399/sending-blocks-of-2d-array-in-c-using-mpi/9271753#9271753
    MPI_Datatype coltype, resizedcoltype;

    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &coltype);

    MPI_Type_commit(&coltype);

    MPI_Type_create_resized(coltype, 0, 1 * sizeof(double), &resizedcoltype); // set upper bound of the datastrcture to be just one double away from the last one, sth. the next collumn starts just one element from the previus one
    MPI_Type_commit(&resizedcoltype);

    int *displsMatrixTranspose = (int *)malloc(size * sizeof(int));  // stores start index of each block
    int *scountsMatrixTranspose = (int *)malloc(size * sizeof(int)); // stores number of elements in each block
    int block_start = 0;
    for (int i = 0; i < size; i++)
    {
        if (i < size - 1)
        {
            scountsMatrixTranspose[i] = blockSize;
        }
        else
        {
            scountsMatrixTranspose[i] = (blockSize + remSize);
        }
        displsMatrixTranspose[i] = block_start;
        block_start += scountsMatrixTranspose[i];
    }

    MPI_Request send_req_A_result_transpose, send_req_z, send_req_x;

    // transpose by distributing localSize columns as rows
    double *local_A_result_transpose = (double *)malloc(localSize * n * sizeof(double));
    MPI_Wait(&recv_req_A_result, MPI_STATUS_IGNORE);
    MPI_Iscatterv(A_result, scountsMatrixTranspose, displsMatrixTranspose, resizedcoltype, local_A_result_transpose, localSize * n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_req_A_result_transpose);

    // send vectors z,x : all processes get size: blockSize, except last process which gets the remaining elements
    double *local_z = (double *)malloc(localSize * sizeof(double));
    double *local_x = (double *)malloc(localSize * sizeof(double));

    MPI_Iscatterv(z, scountsVector, displsVector, MPI_DOUBLE, local_z, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_req_z);
    MPI_Iscatterv(x, scountsVector, displsVector, MPI_DOUBLE, local_x, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_req_x);

    // send y entirely to all processes
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    // Wait for the non-blocking communication to complete
    MPI_Wait(&send_req_A_result_transpose, MPI_STATUS_IGNORE);
    MPI_Wait(&send_req_z, MPI_STATUS_IGNORE);
    MPI_Wait(&send_req_x, MPI_STATUS_IGNORE);
    // compute x = beta * (A_result)T * y + z
    /*
    for (int i = 0; i < localSize; i++)
    {
        for (int j = 0; j < n; j++)
        {
            local_x[i] = local_x[i] + beta * local_A_result_transpose[i * n + j] * y[j];
        }
        local_x[i] += local_z[i];
    }    
    */


    // compute x = beta * (A_result)T * y + z
    part_2_2(localSize, n, beta, local_A_result_transpose, local_x, y, local_z);
    // Non-blocking communication for gathering x_result
    MPI_Request recv_req_x_result;
    // recieve x_result
    MPI_Igatherv(local_x, localSize, MPI_DOUBLE, x_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD, &recv_req_x_result);
    
    /*
    Part 3: w_result = alpha * A_result * x_result
    */
    MPI_Request send_req_A_result, send_req_x_result, send_req_w;

    // send A_result_transpose: all processes get blockSize rows, except last process which gets the remaining elements
    double *local_A_result = (double *)malloc(localSize * n * sizeof(double));
    MPI_Iscatterv(A_result, scountsMatrix, displsMatrix, MPI_DOUBLE, local_A_result, localSize * n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_req_A_result);

    // send x entirely to all processes
    MPI_Wait(&recv_req_x_result, MPI_STATUS_IGNORE);
    MPI_Ibcast(x_result, n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_req_x_result);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // send w : all processes get size: blockSize, except last process which gets the remaining elements
    double *local_w = (double *)malloc(localSize * sizeof(double));
    MPI_Iscatterv(w, scountsVector, displsVector, MPI_DOUBLE, local_w, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_req_w);

    // Wait for the non-blocking communication to complete
    MPI_Wait(&send_req_A_result, MPI_STATUS_IGNORE);
    MPI_Wait(&send_req_x_result, MPI_STATUS_IGNORE);
    
    MPI_Wait(&send_req_w, MPI_STATUS_IGNORE);
    // compute w = alpha * A_result * x_result
    /*
    for (int i = 0; i < localSize; i++)
        for (int j = 0; j < n; j++)
            local_w[i] = local_w[i] + alpha * local_A_result[i * n + j] * x_result[j];  
    */
    
    part_3(localSize, n, alpha, local_A_result, x_result, local_w);

    // recieve w_result
    MPI_Request recv_req_w_result;
    MPI_Igatherv(local_w, localSize, MPI_DOUBLE, w_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD, &recv_req_w_result);
    MPI_Wait(&recv_req_w_result, MPI_STATUS_IGNORE);
    
    // MPI_Free_mem temporary arrays
    MPI_Free_mem(displsVector);
    MPI_Free_mem(scountsVector);
    MPI_Free_mem(displsMatrix);
    MPI_Free_mem(scountsMatrix);
    MPI_Free_mem(local_u1);
    MPI_Free_mem(local_u2);
    MPI_Free_mem(local_A);
    MPI_Free_mem(local_A_result_transpose);
    MPI_Free_mem(local_z);
    MPI_Free_mem(local_x);
    MPI_Free_mem(local_w);
}