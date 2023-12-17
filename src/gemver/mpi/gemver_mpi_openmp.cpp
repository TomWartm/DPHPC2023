#include <mpi.h>
#include "omp.h"
#include <iomanip>
#include "../../helpers/gemver_init.h"
#include <iostream>
#include <chrono>


using namespace std::chrono;
using namespace std;




void gemver_mpi_3_new_openmp(int n,  double *A_result, double *x_result, double *w_result)
{   

    omp_set_num_threads(NUM_THREADS);
    //std::cout <<"NumThreads: "<< numThreads << std::endl;
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
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < localSize; j++)
        {
            local_A[i * localSize + j] += u1[i] * local_v1[j] + u2[i] * local_v2[j];
        }
    }
    
    
    // get local results back to process 0
    MPI_Gatherv(local_A, localSize , resizedcoltype, A_result, scountsMatrixTranspose, displsMatrixTranspose, column_recv_type1 , 0, MPI_COMM_WORLD); // but that has no hurry (can finish in the end)
    
    


    /*
    Part 2: x_result = x + beta * (A_result)T * y + z
    */

    // compute x = beta * (A_result)T * y + z

    #pragma omp parallel for
    for (int j = 0; j < localSize; j++){
        for (int i = 0; i < n; i++){
        
            local_x[j] = local_x[j] + beta * local_A[i * localSize + j] * y[i]; // transpose A in here 
        }
        
    }
    #pragma omp parallel for
    for (int j = 0; j < localSize; j++){
        local_x[j] += local_z[j];
    }
       
    // recieve x_result on process 0
    MPI_Gatherv(local_x, localSize, MPI_DOUBLE, x_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD); // but that has no hurry (can finish in the end)


    // compute w = alpha * A_result * x_result
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < localSize; j++)
            local_w[i] = local_w[i] + alpha * local_A[i * localSize + j] * local_x[j];



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