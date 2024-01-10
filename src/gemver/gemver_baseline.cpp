#include <iostream>
#include <cmath>
#include <cblas.h>

void kernel_gemver(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{

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

void kernel_gemver_openblas(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    cblas_dger(CblasRowMajor, n, n, 1.0, u1, 1, v1, 1, A, n);
    cblas_dger(CblasRowMajor, n, n, 1.0, u2, 1, v2, 1, A, n);

    cblas_dgemv(CblasRowMajor, CblasTrans, n, n, beta, A, n, y, 1, 1.0, x, 1);
    cblas_daxpy(n, 1.0, z, 1, x, 1);

    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, alpha, A, n, x, 1, 1.0, w, 1);
}

void kernel_gemver(int n, int m, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            A[i * m + j] = A[i * m + j] + u1[i] * v1[j] + u2[i] * v2[j];

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            x[i] = x[i] + beta * A[j * m + i] * y[j];

    for (int i = 0; i < m; i++)
        x[i] = x[i] + z[i];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            w[i] = w[i] + alpha * A[i * m + j] * x[j];
}
/*
    Blocked versions 
*/

void part_1(int n, int m, double *A, double *u1, double *u2, double *v1, double *v2){
    /*
    Part1:
    A += u1*v1 + u2*v2
    n: rows
    m: columns
    */


    const int blockSize = 4; // You can adjust the blockSize based on your system and matrix size
    int i_max = floor(n/blockSize)* blockSize;
    int j_max = floor(m/blockSize)* blockSize;
    int i, j;


    double u1_0, u1_1, u1_2, u1_3;
    double v1_0, v1_1, v1_2, v1_3;
    double u2_0, u2_1, u2_2, u2_3;
    double v2_0, v2_1, v2_2, v2_3;  
    // Process main blocks
    for (i = 0; i < i_max; i += blockSize) {
        for (j = 0; j < j_max ; j += blockSize) {

            u1_0 = u1[(i + 0)];
            u1_1 = u1[(i + 1)];
            u1_2 = u1[(i + 2)];
            u1_3 = u1[(i + 3)];

            u2_0 = u2[(i + 0)];
            u2_1 = u2[(i + 1)];
            u2_2 = u2[(i + 2)];
            u2_3 = u2[(i + 3)];

            v1_0 = v1[(j + 0)];
            v1_1 = v1[(j + 1)];
            v1_2 = v1[(j + 2)];
            v1_3 = v1[(j + 3)];

            v2_0 = v2[(j + 0)];
            v2_1 = v2[(j + 1)];
            v2_2 = v2[(j + 2)];
            v2_3 = v2[(j + 3)];

            A[(i + 0)*m +(j + 0)] += u1_0 * v1_0 + u2_0 * v2_0;
            A[(i + 0)*m +(j + 1)] += u1_0 * v1_1 + u2_0 * v2_1;
            A[(i + 0)*m +(j + 2)] += u1_0 * v1_2 + u2_0 * v2_2;
            A[(i + 0)*m +(j + 3)] += u1_0 * v1_3 + u2_0 * v2_3;

            A[(i + 1)*m +(j + 0)] += u1_1 * v1_0 + u2_1 * v2_0;
            A[(i + 1)*m +(j + 1)] += u1_1 * v1_1 + u2_1 * v2_1;
            A[(i + 1)*m +(j + 2)] += u1_1 * v1_2 + u2_1 * v2_2;
            A[(i + 1)*m +(j + 3)] += u1_1 * v1_3 + u2_1 * v2_3;

            A[(i + 2)*m +(j + 0)] += u1_2 * v1_0 + u2_2 * v2_0;
            A[(i + 2)*m +(j + 1)] += u1_2 * v1_1 + u2_2 * v2_1;
            A[(i + 2)*m +(j + 2)] += u1_2 * v1_2 + u2_2 * v2_2;
            A[(i + 2)*m +(j + 3)] += u1_2 * v1_3 + u2_2 * v2_3;

            A[(i + 3)*m +(j + 0)] += u1_3 * v1_0 + u2_3 * v2_0;
            A[(i + 3)*m +(j + 1)] += u1_3 * v1_1 + u2_3 * v2_1;
            A[(i + 3)*m +(j + 2)] += u1_3 * v1_2 + u2_3 * v2_2;
            A[(i + 3)*m +(j + 3)] += u1_3 * v1_3 + u2_3 * v2_3;


        }
    }
    // Process remaining elements in rows
    for (i = i_max; i < n; i++) {
        for (j = 0; j < j_max; j++) {
            A[i * m + j] += u1[i] * v1[j] + u2[i] * v2[j];
            
        }
    }
    // Process remaining elements in columns
    for (i = 0; i < n; i++) {
        for (j = j_max; j < m; j++) {
            A[i * m + j] += u1[i] * v1[j] + u2[i] * v2[j];
            
        }
    }
}

void part_2(int n, int m, double beta, double *A, double *x, double *y, double *z){
/*  
    Part 2: 
    x += beta * (A)T * y
    x += z
    
    Reduces multiplication with beta by factor of 4 (blockSize)
    Reduces Memory reads by a factor of 4
    n: rows
    m: columns
    */


    const int blockSize = 4; // You can adjust the blockSize based on your system and matrix size
    int i_max = floor(n/blockSize)* blockSize;
    int j_max = floor(m/blockSize)* blockSize;
    int i, j;


    /*
    Part 1.1
    
    x += beta * (A)T * y
    
    */
    double y_0, y_1, y_2, y_3;

    // Process main blocks
    for (i = 0; i < i_max; i += blockSize) {
        for (j = 0; j < j_max ; j += blockSize) {
            y_0 = y[(j + 0)];
            y_1 = y[(j + 1)];
            y_2 = y[(j + 2)];
            y_3 = y[(j + 3)];

            x[(i + 0)] = x[(i + 0)] + beta * (A[(j + 0) * n + (i + 0)] * y_0 + A[(j + 1) * n + (i + 0)] * y_1 + A[(j + 2) * n + (i + 0)] * y_2 + A[(j + 3) * n + (i + 0)] * y_3);

            x[(i + 1)] = x[(i + 1)] + beta * (A[(j + 0) * n + (i + 1)] * y_0 + A[(j + 1) * n + (i + 1)] * y_1 + A[(j + 2) * n + (i + 1)] * y_2 + A[(j + 3) * n + (i + 1)] * y_3);

            x[(i + 2)] = x[(i + 2)] + beta * (A[(j + 0) * n + (i + 2)] * y_0 + A[(j + 1) * n + (i + 2)] * y_1 + A[(j + 2) * n + (i + 2)] * y_2 + A[(j + 3) * n + (i + 2)] * y_3);

            x[(i + 3)] = x[(i + 3)] + beta * (A[(j + 0) * n + (i + 3)] * y_0 + A[(j + 1) * n + (i + 3)] * y_1 + A[(j + 2) * n + (i + 3)] * y_2 + A[(j + 3) * n + (i + 3)] * y_3);


        }
    }

    // Process remaining elements in rows
    for (i = i_max; i < n; i++) {
        for (j = 0; j < j_max; j++) {
            x[i] = x[i] + beta * A[j * n + i] * y[j];
            
        }
    }
    // Process remaining elements in columns
    for (i = 0; i < n; i++) {
        for (j = j_max; j < m; j++) {
            x[i] = x[i] + beta * A[j * n + i] * y[j];
            
        }
    }



    /*
    Part 2.1

    x += z
    */
    
    for (int i = 0; i < i_max; i+= blockSize){
        x[i + 0] += z[i + 0];
        x[i + 1] += z[i + 1];
        x[i + 2] += z[i + 2];
        x[i + 3] += z[i + 3];
    }
       
    // Process remaining elements in rows
    for (i = i_max; i < n; i++) {
          
        x[i] += z[i];
            
    }
}

void part_3(int n, int m, double alpha, double *A, double *x,  double *w ){

    /*
    Part 3:
    w += alpha * A * x

    Reduces Memory reads of x, y by a factor of 4
    Reduces Multiplications with alpha by a factor of 4
    */

    const int blockSize = 4; // You can adjust the blockSize based on your system and matrix size
    int i_max = floor(n/blockSize)* blockSize;
    int j_max = floor(m/blockSize)* blockSize;
    int i, j;


    double x_0, x_1, x_2, x_3;

    // Process main blocks
    for (i = 0; i < i_max; i += blockSize) {
        for (j = 0; j < j_max ; j += blockSize) {
            
            x_0 = x[j + 0];
            x_1 = x[j + 1];
            x_2 = x[j + 2];
            x_3 = x[j + 3];
            
            w[i + 0] += alpha * (A[(i + 0) * m + (j + 0)] * x_0 + A[(i + 0) * m + (j + 1)] * x_1 + A[(i + 0) * m + (j + 2)] * x_2 + A[(i + 0) * m + (j + 3)] * x_3);
            w[i + 1] += alpha * (A[(i + 1) * m + (j + 0)] * x_0 + A[(i + 1) * m + (j + 1)] * x_1 + A[(i + 1) * m + (j + 2)] * x_2 + A[(i + 1) * m + (j + 3)] * x_3);
            w[i + 2] += alpha * (A[(i + 2) * m + (j + 0)] * x_0 + A[(i + 2) * m + (j + 1)] * x_1 + A[(i + 2) * m + (j + 2)] * x_2 + A[(i + 2) * m + (j + 3)] * x_3);
            w[i + 3] += alpha * (A[(i + 3) * m + (j + 0)] * x_0 + A[(i + 3) * m + (j + 1)] * x_1 + A[(i + 3) * m + (j + 2)] * x_2 + A[(i + 3) * m + (j + 3)] * x_3);


        }
    }

    // Process remaining elements in rows
    for (i = i_max; i < n; i++) {
        for (j = 0; j < j_max; j++) {
            w[i] = w[i] + alpha * A[i * m + j] * x[j];
            //std::cout << "stop1" <<std::endl;
        }
    }
    // Process remaining elements in columns
    for (i = 0; i < n; i++) {
        for (j = j_max; j < m; j++) {
            w[i] = w[i] + alpha * A[i * m + j] * x[j];
            //std::cout << "stop2" <<std::endl;
        }
    }

}



void gemver_baseline_blocked_1(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{

    part_1(n, n, A, u1, u2, v1, v2);

    part_2(n, n, beta, A, x, y, z);

    part_3(n, n, alpha, A, x, w);
}



void part_2_2(int n, int m, double beta, double *A_transposed, double *x, double *y, double *z){
/*  
    Part 2: 
    x += beta * (A)T * y
    x += z
    
    same as part_2_1 but recieves already transposed matrix ( this is for MPI where transposition happens wen distributing data)
    n: rows
    m: columns
    */


    const int blockSize = 4; // You can adjust the blockSize based on your system and matrix size
    int i_max = floor(n/blockSize)* blockSize;
    int j_max = floor(m/blockSize)* blockSize;
    int i, j;
    

    /*
    Part 1.1
    
    x += beta * (A)T * y
    
    */
    double y_0, y_1, y_2, y_3;

    // Process main blocks
    for (i = 0; i < i_max; i += blockSize) {
        for (j = 0; j < j_max ; j += blockSize) {
            y_0 = y[(j + 0)];
            y_1 = y[(j + 1)];
            y_2 = y[(j + 2)];
            y_3 = y[(j + 3)];

            x[(i + 0)] = x[(i + 0)] + beta * (A_transposed[(i + 0) * m + (j + 0)] * y_0 + A_transposed[(i + 0) * m + (j + 1)] * y_1 + A_transposed[(i + 0) * m + (j + 2)] * y_2 + A_transposed[(i + 0) * m + (j + 3)] * y_3);

            x[(i + 1)] = x[(i + 1)] + beta * (A_transposed[(i + 1) * m + (j + 0)] * y_0 + A_transposed[(i + 1) * m + (j + 1)] * y_1 + A_transposed[(i + 1) * m + (j + 2)] * y_2 + A_transposed[(i + 1) * m + (j + 3)] * y_3);

            x[(i + 2)] = x[(i + 2)] + beta * (A_transposed[(i + 2) * m + (j + 0)] * y_0 + A_transposed[(i + 2) * m + (j + 1)] * y_1 + A_transposed[(i + 2) * m + (j + 2)] * y_2 + A_transposed[(i + 2) * m + (j + 3)] * y_3);

            x[(i + 3)] = x[(i + 3)] + beta * (A_transposed[(i + 3) * m + (j + 0)] * y_0 + A_transposed[(i + 3) * m + (j + 1)] * y_1 + A_transposed[(i + 3) * m + (j + 2)] * y_2 + A_transposed[(i + 3) * m + (j + 3)] * y_3);

            
        }
    }

    // Process remaining elements in rows
    for (i = i_max; i < n; i++) {
        for (j = 0; j < j_max; j++) {
            x[i] = x[i] + beta * A_transposed[i * m + j] * y[j];
        }
    }
    // Process remaining elements in columns
    for (i = 0; i < n; i++) {
        for (j = j_max; j < m; j++) {
            x[i] = x[i] + beta * A_transposed[i * m + j] * y[j];
            
        }
    }



    /*
    Part 2.1

    x += z
    */
    
    for (int i = 0; i < i_max; i+= blockSize){
        x[i + 0] += z[i + 0];
        x[i + 1] += z[i + 1];
        x[i + 2] += z[i + 2];
        x[i + 3] += z[i + 3];
    }
       
    // Process remaining elements in rows
    for (i = i_max; i < n; i++) {
          
        x[i] += z[i];
            
    }
}



void gemver_baseline_blocked_2(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{   
    /*
    square
    */

    part_1(n, n, A, u1, u2, v1, v2);

    double *A_transposed = (double *)malloc((n * n) * sizeof(double)); 

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A_transposed[j * n + i] = A[i * n + j];
        }
    }
    
    part_2_2(n, n, beta, A_transposed, x, y, z);
    part_3(n, n, alpha, A, x, w);


    free(A_transposed);
}

void gemver_baseline_blocked_2(int n, int m,  double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    /*
    non-square
    */

    part_1(n, m, A, u1, u2, v1, v2);

    double *A_transposed = (double *)malloc((m * n) * sizeof(double)); 

    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            A_transposed[j * n + i] = A[i * m + j];
        }
    }
    
    part_2_2(m, n, beta, A_transposed, x, y, z);
    part_3(n, m, alpha, A, x, w);


    free(A_transposed);
}