#include <mpi.h>
#include <iomanip>
#include "omp.h"
#include "../gemver_baseline.h"
#include "../../helpers/gemver_init.h"

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

void gemver_mpi_2_new_blocking(int n,  double *A_result, double *x_result, double *w_result)
{   

    /*
    Works for all numbers of processes.
    Same as gemver_mpi_2 but initializes data on each node.
    */
    double alpha; 
    double beta;
    double *v1; 
    double *v2; 
    double *y;
    double *x;



    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // assign memory to variables that are used on all processes
    // A_result, x_result, w_result are initialized on all processes outside of this function
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v1);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v2);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &y);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);

    // initialize data that are on all processes
    init_gemver_vy(n, &alpha, &beta, v1, v2, y);


    // split u1, u2 and A into blocks and distribute over all processes
    int blockSize = n / size;
    int remSize = n % size;
    
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


    /// initialize data
    double *local_A = (double *)malloc(localSize * n * sizeof(double));
    double *local_u1 = (double *)malloc(localSize * sizeof(double));
    double *local_u2 = (double *)malloc(localSize * sizeof(double));
    
    double *local_z = (double *)malloc(localSize * sizeof(double));
    double *local_x = (double *)malloc(localSize * sizeof(double));
    double *local_w = (double *)malloc(localSize * sizeof(double));

    double *local_A_result_transpose = (double *)malloc(localSize * n * sizeof(double));

    init_gemver_xz(n, displsVector[rank], displsVector[rank] + localSize, local_x, local_z);   
    init_gemver_Au(n, n, displsVector[rank], displsVector[rank]+localSize, local_A, local_u1, local_u2);
    init_gemver_w(localSize, local_w);

    
    /*
    Part 1: A_result = A + u1*v1 + u2*v2
    */
    // for (int i = 0; i < localSize; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         local_A[i * n + j] += local_u1[i] * v1[j] + local_u2[i] * v2[j];
    //     }
    // }
    part_1(localSize, n, local_A, local_u1, local_u2, v1, v2);
    
    // get local results back to process 0
    MPI_Gatherv(local_A, localSize * n, MPI_DOUBLE, A_result, scountsMatrix, displsMatrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    /*
    Part 2: x_result = x + beta * (A_result)T * y + z
    */

    // transpose by distributing localSize columns as rows
    MPI_Scatterv(A_result, scountsMatrixTranspose, displsMatrixTranspose, resizedcoltype, local_A_result_transpose, localSize * n, MPI_DOUBLE, 0, MPI_COMM_WORLD); // A_result -> local_A_result_transpose

    // compute x = beta * (A_result)T * y + z

    // for (int i = 0; i < localSize; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         local_x[i] = local_x[i] + beta * local_A_result_transpose[i * n + j] * y[j];
    //     }
    //     local_x[i] += local_z[i];
    // }

    part_2_2(localSize, n, beta, local_A_result_transpose, local_x, y, local_z);

    // recieve x_result adn send to all processes
    MPI_Allgatherv(local_x, localSize, MPI_DOUBLE, x_result, scountsVector, displsVector, MPI_DOUBLE, MPI_COMM_WORLD);

    // compute w = alpha * A_result * x_result
    // for (int i = 0; i < localSize; i++)
    //     for (int j = 0; j < n; j++)
    //         local_w[i] = local_w[i] + alpha * local_A[i * n + j] * x_result[j];

    part_3(localSize, n, alpha, local_A, x_result, local_w);

    // recieve w_result
    MPI_Gatherv(local_w, localSize, MPI_DOUBLE, w_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    
    // MPI_Free_mem temporary arrays
    MPI_Free_mem(displsVector);
    MPI_Free_mem(scountsVector);
    MPI_Free_mem(displsMatrix);
    MPI_Free_mem(scountsMatrix);
    MPI_Free_mem(displsMatrixTranspose);
    MPI_Free_mem(scountsMatrixTranspose);    
    MPI_Free_mem(local_u1);
    MPI_Free_mem(local_u2);
    MPI_Free_mem(local_A);
    MPI_Free_mem(local_A_result_transpose);
    
    MPI_Free_mem(local_z);
    MPI_Free_mem(local_x);
    MPI_Free_mem(local_w);
    MPI_Free_mem(v1);
    MPI_Free_mem(v2);
    MPI_Free_mem(y);
    MPI_Free_mem(x);



}



void gemver_mpi_2_new_openmp(int n,  double *A_result, double *x_result, double *w_result)
{
    /*
    Works for all numbers of processes.
    Same as gemver_mpi_2 but initializes data on each node.
    */
    double alpha; 
    double beta;
    double *v1; 
    double *v2; 
    double *y;
    double *x;



    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // assign memory to variables that are used on all processes
    // A_result, x_result, w_result are initialized on all processes outside of this function
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v1);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &v2);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &y);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);

    // initialize data that are on all processes
    init_gemver_vy(n, &alpha, &beta, v1, v2, y);


    // split u1, u2 and A into blocks and distribute over all processes
    int blockSize = n / size;
    int remSize = n % size;
    
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


    /// initialize data
    double *local_A = (double *)malloc(localSize * n * sizeof(double));
    double *local_u1 = (double *)malloc(localSize * sizeof(double));
    double *local_u2 = (double *)malloc(localSize * sizeof(double));
    
    double *local_z = (double *)malloc(localSize * sizeof(double));
    double *local_x = (double *)malloc(localSize * sizeof(double));
    double *local_w = (double *)malloc(localSize * sizeof(double));

    double *local_A_result_transpose = (double *)malloc(localSize * n * sizeof(double));

    init_gemver_xz(n, displsVector[rank], displsVector[rank] + localSize, local_x, local_z);   
    init_gemver_Au(n, n, displsVector[rank], displsVector[rank]+localSize, local_A, local_u1, local_u2);
    init_gemver_w(localSize, local_w);

    
    /*
    Part 1: A_result = A + u1*v1 + u2*v2
    */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < localSize; i++)
    {
        for (int j = 0; j < n; j++)
        {
            local_A[i * n + j] += local_u1[i] * v1[j] + local_u2[i] * v2[j];
        }
    
    }
    //part_1(localSize, n, local_A, local_u1, local_u2, v1, v2);
    
    // get local results back to process 0
    MPI_Gatherv(local_A, localSize * n, MPI_DOUBLE, A_result, scountsMatrix, displsMatrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    /*
    Part 2: x_result = x + beta * (A_result)T * y + z
    */

    // transpose by distributing localSize columns as rows
    MPI_Scatterv(A_result, scountsMatrixTranspose, displsMatrixTranspose, resizedcoltype, local_A_result_transpose, localSize * n, MPI_DOUBLE, 0, MPI_COMM_WORLD); // A_result -> local_A_result_transpose

    // compute x = beta * (A_result)T * y + z
    #pragma omp parallel for
    for (int i = 0; i < localSize; i++)
    {
        for (int j = 0; j < n; j++)
        {
            local_x[i] = local_x[i] + beta * local_A_result_transpose[i * n + j] * y[j];
        }
        local_x[i] += local_z[i];
    }
    
    //part_2_2(localSize, n, beta, local_A_result_transpose, local_x, y, local_z);

    // recieve x_result adn send to all processes
    MPI_Allgatherv(local_x, localSize, MPI_DOUBLE, x_result, scountsVector, displsVector, MPI_DOUBLE, MPI_COMM_WORLD);
    
    //compute w = alpha * A_result * x_result
    #pragma omp parallel for 
    for (int i = 0; i < localSize; i++)
        for (int j = 0; j < n; j++)
            local_w[i] = local_w[i] + alpha * local_A[i * n + j] * x_result[j];

    
    //part_3(localSize, n, alpha, local_A, x_result, local_w);

    // recieve w_result
    MPI_Gatherv(local_w, localSize, MPI_DOUBLE, w_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    
    // MPI_Free_mem temporary arrays
    MPI_Free_mem(displsVector);
    MPI_Free_mem(scountsVector);
    MPI_Free_mem(displsMatrix);
    MPI_Free_mem(scountsMatrix);
    MPI_Free_mem(displsMatrixTranspose);
    MPI_Free_mem(scountsMatrixTranspose);    
    MPI_Free_mem(local_u1);
    MPI_Free_mem(local_u2);
    MPI_Free_mem(local_A);
    MPI_Free_mem(local_A_result_transpose);
    
    MPI_Free_mem(local_z);
    MPI_Free_mem(local_x);
    MPI_Free_mem(local_w);
    MPI_Free_mem(v1);
    MPI_Free_mem(v2);
    MPI_Free_mem(y);
    MPI_Free_mem(x);



}


void gemver_mpi_3_new_blocking(int n,  double *A_result, double *x_result, double *w_result)
{   
    //auto start = high_resolution_clock::now();

    /*
        A is splited into columns instead of rows. This leads to one fewer Scatter, because transposition is done inplace, but leads to one reduce in the end that is very slow. 
    */
    double alpha; 
    double beta;
    double *u1; 
    double *u2; 
    double *y;
    double *x;



    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // assign memory to variables that are used on all processes
    // A_result, x_result, w_result are initialized on all processes outside of this function
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &u1);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &u2);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &y);
    MPI_Alloc_mem(n * sizeof(double), MPI_INFO_NULL, &x);

    // initialize data that are on all processes
    init_gemver_uy(n, &alpha, &beta, u1, u2, y);


    // split u1, u2 and A into blocks and distribute over all processes
    int blockSize = n / size;
    int remSize = n % size;
    
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

    
    MPI_Datatype coltype, resizedcoltype;

    MPI_Type_vector(n, 1, localSize, MPI_DOUBLE, &coltype);

    MPI_Type_commit(&coltype);

    MPI_Type_create_resized(coltype, 0, 1 * sizeof(double), &resizedcoltype); // set upper bound of the datastrcture to be just one double away from the last one, sth. the next collumn starts just one element from the previus one
    MPI_Type_commit(&resizedcoltype);


    MPI_Datatype column_recv_type,column_recv_type1;
    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &column_recv_type);
    MPI_Type_commit(&column_recv_type);

    MPI_Type_create_resized(column_recv_type, 0, sizeof(double), &column_recv_type1);
    MPI_Type_commit(&column_recv_type1);


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


    /// initialize data
    double *local_A = (double *)malloc(localSize * n * sizeof(double));
    double *local_v1 = (double *)malloc(localSize * sizeof(double));
    double *local_v2 = (double *)malloc(localSize * sizeof(double));
    
    double *local_z = (double *)malloc(localSize * sizeof(double));
    double *local_x = (double *)malloc(localSize * sizeof(double));
    double *local_w = (double *)malloc(n * sizeof(double));

    double *local_A_result_transpose = (double *)malloc(localSize * n * sizeof(double));

    init_gemver_xz(n, displsVector[rank], displsVector[rank] + localSize, local_x, local_z);   
    init_gemver_Av(n, n, displsVector[rank], displsVector[rank]+localSize, local_A, local_v1, local_v2);
    init_gemver_w(n, local_w);
    
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
 
    // cout << "Rank: "<<rank <<" Initialization: " << duration.count()/1000 << " ms" << endl;
    // start = high_resolution_clock::now();
    
    
    /*
    Part 1: A_result = A + u1*v1 + u2*v2
    */
    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < localSize; j++)
    //     {
    //         local_A[i * localSize + j] += u1[i] * local_v1[j] + u2[i] * local_v2[j];
    //     }
    // }

    part_1(n, localSize, local_A, u1, u2, local_v1, local_v2);
    
    // get local results back to process 0
    MPI_Gatherv(local_A, localSize , resizedcoltype, A_result, scountsMatrixTranspose, displsMatrixTranspose, column_recv_type1 , 0, MPI_COMM_WORLD); // but that has no hurry (can finish in the end)
    
;     


    /*
    Part 2: x_result = x + beta * (A_result)T * y + z
    */

    // compute x = beta * (A_result)T * y + z

    
    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < localSize; j++)
    //     {
    //         local_x[j] = local_x[j] +  local_A[i * localSize + j] * y[i]; // transpose A in here 
    //     }
        
    // }
    // // move multiplication with beta out of doubbled loop
    // for (int j = 0; j < localSize; j++){
    //     local_x[j] = local_x[j]*beta + local_z[j];
    // }
    part_2(localSize, n, beta, local_A, local_x, y, local_z);
    // recieve x_result on process 0
    MPI_Gatherv(local_x, localSize, MPI_DOUBLE, x_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD); // but that has no hurry (can finish in the end)


    // compute w = alpha * A_result * x_result
    // for (int i = 0; i < n; i++)
    //     for (int j = 0; j < localSize; j++)
    //         local_w[i] = local_w[i] + alpha * local_A[i * localSize + j] * local_x[j];

    // compute w = alpha * A_result * x_result
    // for (int i = 0; i < localSize; i++)
    //     for (int j = 0; j < n; j++)
    //         local_w[i] = local_w[i] + alpha * local_A[i * n + j] * x_result[j];

    part_3(n, localSize, alpha, local_A, local_x, local_w);


    // add w_result from each process to w_result of process 0
    MPI_Reduce(local_w, w_result, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // MPI_Free_mem temporary arrays
    MPI_Free_mem(displsVector);
    MPI_Free_mem(scountsVector);
    MPI_Free_mem(displsMatrixTranspose);
    MPI_Free_mem(scountsMatrixTranspose);    
    MPI_Free_mem(local_v1);
    MPI_Free_mem(local_v2);
    MPI_Free_mem(local_A);
    MPI_Free_mem(local_A_result_transpose);
    
    MPI_Free_mem(local_z);
    MPI_Free_mem(local_x);
    MPI_Free_mem(local_w);
    MPI_Free_mem(u1);
    MPI_Free_mem(u2);
    MPI_Free_mem(y);
    MPI_Free_mem(x);



}