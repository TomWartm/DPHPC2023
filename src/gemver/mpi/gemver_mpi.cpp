#include <mpi.h>
#include <iomanip>

template <typename T>
void printVector(std::string var_name, int size, T* arr) {
    std::cout << var_name << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}
void printArray(std::string var_name, int size_n, int size_m, double* arr) {
    const int fieldWidth = 5; // Adjust the width as needed
    std::cout << var_name << std::endl;
    for (int i = 0; i < size_n; ++i) {
        for (int j = 0; j < size_m; ++j) {
            std::cout << std::setw(fieldWidth) << arr[i*size_n+j] << " ";
        }
        std::cout << std::endl;
    }
}

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


void gemver_mpi_2(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z, double *A_result, double *x_result, double *w_result)
{   
    /*
    Main process: 0
    Works for all numbers of processes.
    Main process distributes the work in blocks equally on all processes and collects results in between. 
    */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    


    /* 
    Part 1: A_result = A + u1*v1 + u2*v2
    */

    // split u1, u2 and A into blocks and distribute over all processes 
    int blockSize = n/size;
    int remSize = n%size;
    // send vectors u1, u2: all processes get size: blockSize, except last process which gets the remaining elements
    int* displsVector = (int *)malloc(size*sizeof(int)); // stores start index of each block
    int* scountsVector = (int *)malloc(size*sizeof(int)); // stores number of elements in each block
    for (int i = 0; i < size; i++){
        if (i < size-1){
            scountsVector[i] = blockSize;
        }else{
            scountsVector[i] = blockSize + remSize;
        }
        displsVector[i] = i*blockSize;

    }
    int localSize = scountsVector[rank]; // for al processes < size -1 == blockSize else blockSize + remSize
    double* local_u1 = (double*)malloc(localSize * sizeof(double));
    double* local_u2 = (double*)malloc(localSize * sizeof(double));

    MPI_Scatterv(u1,scountsVector, displsVector, MPI_DOUBLE, local_u1, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // u1 -> local_u1
    MPI_Scatterv(u2,scountsVector, displsVector, MPI_DOUBLE, local_u2, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // u2 -> local_u2
    
    // send array A: all processes get blockSize rows, except last process which gets the remaining elements
    int* displsMatrix = (int *)malloc(size*sizeof(int)); // stores start index of each block
    int* scountsMatrix = (int *)malloc(size*sizeof(int)); // stores number of elements in each block
    for (int i = 0; i < size; i++){
        if (i < size-1){
            scountsMatrix[i] = blockSize*n;
        }else{
            scountsMatrix[i] = (blockSize + remSize)*n;
        }
        displsMatrix[i] = i*blockSize*n;

    }

    
    
    double* local_A = (double*)malloc(localSize*n * sizeof(double));
    MPI_Scatterv(A,scountsMatrix, displsMatrix, MPI_DOUBLE, local_A, localSize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD); // A -> local_A
    
    
    // send v1, v2 entirely to all processes
    MPI_Bcast(v1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    // compute  A += u1 * v1 + u2 * v2
    for (int i = 0; i < localSize; i++){
        for (int j = 0; j < n; j++){
            local_A[i*n + j] += local_u1[i] * v1[j] + local_u2[i] * v2[j];
        }
    }
    // get local results back to process 0
    MPI_Gatherv(local_A, localSize * n, MPI_DOUBLE, A_result, scountsMatrix, displsMatrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

    
    
    /* 
    Part 2: x_result = x + beta * (A_result)T * y + z 
    */




    // transpose  array in process 0
    double* A_result_transpose = (double*)malloc(n*n * sizeof(double));
    
    if (rank==0){
        

        // transpose A_result into A_result_transpose
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A_result_transpose[j*n+i] = A_result[i*n+j];
            }
        }
    }

    

    // send array A_result_transpose: all processes get blockSize rows, except last process which gets the remaining elements
    double* local_A_result_transpose = (double*)malloc(localSize*n * sizeof(double));
    MPI_Scatterv(A_result_transpose,scountsMatrix, displsMatrix, MPI_DOUBLE, local_A_result_transpose, localSize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD); // A_result_transpose -> local_A_result_transpose
    
    // send vectors z,x : all processes get size: blockSize, except last process which gets the remaining elements
    double* local_z = (double*)malloc(localSize * sizeof(double));
    double* local_x = (double*)malloc(localSize * sizeof(double));

    MPI_Scatterv(z,scountsVector, displsVector, MPI_DOUBLE, local_z, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // z -> local_z
    MPI_Scatterv(x,scountsVector, displsVector, MPI_DOUBLE, local_x, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // x -> local_x
   
    // send y entirely to all processes
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // compute x = beta * (A_result)T * y + z 
    
    for (int i = 0; i < localSize; i++){
        for (int j = 0; j < n; j++){
            local_x[i] = local_x[i] + beta * local_A_result_transpose[i * n + j] * y[j];
        }
        local_x[i] +=  local_z[i];
              
    }
        
    
    // recieve x_result
    MPI_Gatherv(local_x, localSize, MPI_DOUBLE, x_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    


    /* 
    Part 3: w_result = alpha * A_result * x_result
    */

   // send A_result_transpose: all processes get blockSize rows, except last process which gets the remaining elements
    double* local_A_result = (double*)malloc(localSize*n * sizeof(double));
    MPI_Scatterv(A_result, scountsMatrix, displsMatrix, MPI_DOUBLE, local_A_result, localSize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD); // A_result -> local_A_result
    
    // send x entirely to all processes
    MPI_Bcast(x_result, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // send w : all processes get size: blockSize, except last process which gets the remaining elements
    double* local_w = (double*)malloc(localSize * sizeof(double));
    MPI_Scatterv(w,scountsVector, displsVector, MPI_DOUBLE, local_w, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // w -> local_w
    

    // compute w = alpha * A_result * x_result
    for (int i = 0; i < localSize; i++)
        for (int j = 0; j < n; j++)
            local_w[i] = local_w[i] + alpha * local_A_result[i * n + j] * x_result[j];

    // recieve w_result
    MPI_Gatherv(local_w, localSize, MPI_DOUBLE, w_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

    // MPI_Free_mem temporary arrays
    MPI_Free_mem(displsVector);
    MPI_Free_mem(scountsVector);
    MPI_Free_mem(displsMatrix);
    MPI_Free_mem(scountsMatrix);
    MPI_Free_mem(local_u1);
    MPI_Free_mem(local_u2);
    MPI_Free_mem(local_A);
    MPI_Free_mem(local_A_result_transpose);
    MPI_Free_mem(A_result_transpose);
    MPI_Free_mem(local_z);
    MPI_Free_mem(local_x);
    MPI_Free_mem(local_w);
 }
void gemver_mpi_3(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z, double *A_result, double *x_result, double *w_result)
{   
    /*
    Works for all numbers of processes.
    Main process distributes the work in blocks equally on all processes and collects results in between.
    matrix transposition is done by Scattering columns as rows to other processes
    */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    


    /* 
    Part 1: A_result = A + u1*v1 + u2*v2
    */

    // split u1, u2 and A into blocks and distribute over all processes 
    int blockSize = n/size;
    int remSize = n%size;
    // send vectors u1, u2: all processes get size: blockSize, except last process which gets the remaining elements
    int* displsVector = (int *)malloc(size*sizeof(int)); // stores start index of each block
    int* scountsVector = (int *)malloc(size*sizeof(int)); // stores number of elements in each block
    for (int i = 0; i < size; i++){
        if (i < size-1){
            scountsVector[i] = blockSize;
        }else{
            scountsVector[i] = blockSize + remSize;
        }
        displsVector[i] = i*blockSize;

    }
    int localSize = scountsVector[rank]; // for al processes < size -1 == blockSize else blockSize + remSize
    double* local_u1 = (double*)malloc(localSize * sizeof(double));
    double* local_u2 = (double*)malloc(localSize * sizeof(double));

    MPI_Scatterv(u1,scountsVector, displsVector, MPI_DOUBLE, local_u1, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // u1 -> local_u1
    MPI_Scatterv(u2,scountsVector, displsVector, MPI_DOUBLE, local_u2, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // u2 -> local_u2
    
    // send array A: all processes get blockSize rows, except last process which gets the remaining elements
    int* displsMatrix = (int *)malloc(size*sizeof(int)); // stores start index of each block
    int* scountsMatrix = (int *)malloc(size*sizeof(int)); // stores number of elements in each block
    for (int i = 0; i < size; i++){
        if (i < size-1){
            scountsMatrix[i] = blockSize*n;
        }else{
            scountsMatrix[i] = (blockSize + remSize)*n;
        }
        displsMatrix[i] = i*blockSize*n;

    }

    
    double* local_A = (double*)malloc(localSize*n * sizeof(double));
    MPI_Scatterv(A,scountsMatrix, displsMatrix, MPI_DOUBLE, local_A, localSize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD); // A -> local_A
    
    
    // send v1, v2 entirely to all processes
    MPI_Bcast(v1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    // compute  A += u1 * v1 + u2 * v2
    for (int i = 0; i < localSize; i++){
        for (int j = 0; j < n; j++){
            local_A[i*n + j] += local_u1[i] * v1[j] + local_u2[i] * v2[j];
        }
    }
    // get local results back to process 0
    MPI_Gatherv(local_A, localSize * n, MPI_DOUBLE, A_result, scountsMatrix, displsMatrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

    
    
    /* 
    Part 2: x_result = x + beta * (A_result)T * y + z 
    */


    // array (A_result)T: all processes get blockSize columns, except last process which gets the remaining elements
    


    // https://stackoverflow.com/questions/9269399/sending-blocks-of-2d-array-in-c-using-mpi/9271753#9271753
    MPI_Datatype coltype, resizedcoltype;

    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &coltype);  

    MPI_Type_commit(&coltype);

    MPI_Type_create_resized(coltype, 0,1*sizeof(double), &resizedcoltype); // set upper bound of the datastrcture to be just one double away from the last one, sth. the next collumn starts just one element from the previus one
    MPI_Type_commit(&resizedcoltype);

    int* displsMatrixTranspose = (int *)malloc(size*sizeof(int)); // stores start index of each block
    int* scountsMatrixTranspose = (int *)malloc(size*sizeof(int)); // stores number of elements in each block
    int block_start = 0;
    for (int i = 0; i < size; i++){
        if (i < size-1){
            scountsMatrixTranspose[i] = blockSize;
        }else{
            scountsMatrixTranspose[i] = (blockSize + remSize);
        }
        displsMatrixTranspose[i] = block_start;
        block_start += scountsMatrixTranspose[i];

    }


    // transpose by distributing localSize columns as rows
    double* local_A_result_transpose = (double*)malloc(localSize*n * sizeof(double));
    MPI_Scatterv(A_result, scountsMatrixTranspose, displsMatrixTranspose, resizedcoltype, local_A_result_transpose, localSize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD );// A_result -> local_A_result_transpose

    // send vectors z,x : all processes get size: blockSize, except last process which gets the remaining elements
    double* local_z = (double*)malloc(localSize * sizeof(double));
    double* local_x = (double*)malloc(localSize * sizeof(double));

    MPI_Scatterv(z,scountsVector, displsVector, MPI_DOUBLE, local_z, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // z -> local_z
    MPI_Scatterv(x,scountsVector, displsVector, MPI_DOUBLE, local_x, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // x -> local_x
   
    // send y entirely to all processes
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // compute x = beta * (A_result)T * y + z 
    
    for (int i = 0; i < localSize; i++){
        for (int j = 0; j < n; j++){
            local_x[i] = local_x[i] + beta * local_A_result_transpose[i * n + j] * y[j];
        }
        local_x[i] += local_z[i];
              
    }
        
    
    // recieve x_result
    MPI_Gatherv(local_x, localSize, MPI_DOUBLE, x_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    


    /* 
    Part 3: w_result = alpha * A_result_transpose * x_result
    */

   // send A_result_transpose: all processes get blockSize rows, except last process which gets the remaining elements
    double* local_A_result = (double*)malloc(localSize*n * sizeof(double));
    MPI_Scatterv(A_result, scountsMatrix, displsMatrix, MPI_DOUBLE, local_A_result, localSize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD); // A_result -> local_A_result
    
    // send x entirely to all processes
    MPI_Bcast(x_result, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // send w : all processes get size: blockSize, except last process which gets the remaining elements
    double* local_w = (double*)malloc(localSize * sizeof(double));
    MPI_Scatterv(w,scountsVector, displsVector, MPI_DOUBLE, local_w, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD); // w -> local_w
    

    // compute w = alpha * A_result * x_result
    for (int i = 0; i < localSize; i++)
        for (int j = 0; j < n; j++)
            local_w[i] = local_w[i] + alpha * local_A_result[i * n + j] * x_result[j];

    // recieve w_result
    MPI_Gatherv(local_w, localSize, MPI_DOUBLE, w_result, scountsVector, displsVector, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

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










