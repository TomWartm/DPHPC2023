#include <stdlib.h>

void init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{

    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double)n;

    for (int i = 0; i < n; i++)
    {
        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        v1[i] = ((i + 1) / fn) / 4.0;
        v2[i] = ((i + 1) / fn) / 6.0;
        y[i] = ((i + 1) / fn) / 8.0;
        z[i] = ((i + 1) / fn) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (int j = 0; j < n; j++)
            A[i * n + j] = (double)(i * j % n) / n;
    }
}

void sparse_init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    // Initialize alpha and beta to some non-zero values
    *alpha = 1.5;
    *beta = 1.2;

    // Initialize vectors with simple non-zero values
    for (int i = 0; i < n; i++)
    {
        u1[i] = i + 1;
        v1[i] = (i + 1) * 2;
        u2[i] = (i + 1) * 3;
        v2[i] = (i + 1) * 4;
        y[i] = (i + 1) * 5;
        z[i] = (i + 1) * 6;
        x[i] = (i + 1) * 7;
        w[i] = (i + 1) * 8;
    }

    // Initialize the matrix A as a tridiagonal matrix
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                A[i * n + j] = (i + 1) * 10; // Main diagonal
            }
            else if (i == j - 1)
            {
                A[i * n + j] = (i + 1) * 5; // Diagonal above the main diagonal
            }
            else if (i == j + 1)
            {
                A[i * n + j] = (i + 1) * 5; // Diagonal below the main diagonal
            }
            else
            {
                A[i * n + j] = 0.0; // All other elements
            }
        }
    }
}

void rand_init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{

    srand(42);

    // Scaling factors
    double alpha_scale = 20.0, alpha_offset = -10.0;
    double beta_scale = 20.0, beta_offset = -10.0;

    // Matrix and vector values
    double A_scale = 2.0, A_offset = -1.0;

    *alpha = (double)rand() / RAND_MAX * alpha_scale + alpha_offset;
    *beta = (double)rand() / RAND_MAX * beta_scale + beta_offset;

    for (int i = 0; i < n; i++)
    {
        u1[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        u2[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        v1[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        v2[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        y[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        z[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        x[i] = 0.0; // Start with zero
        w[i] = 0.0; // Start with zero
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = (double)rand() / RAND_MAX * A_scale + A_offset;
        }
    }
}

void init_gemver(int n, int m, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    // init version of not square matrix
    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double)n;

    for (int i = 0; i < n; i++)
    {
        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        w[i] = 0.0;

        y[i] = ((i + 1) / fn) / 8.0;

        for (int j = 0; j < m; j++)
            A[i * m + j] = (double)(i * j % n) / n;
    }

    for (int j = 0; j < m; j++)
    {
        v1[j] = ((j + 1) / fn) / 4.0;
        v2[j] = ((j + 1) / fn) / 6.0;
        z[j] = ((j + 1) / fn) / 9.0;
        x[j] = 0.0;
    }
}
void init_gemver_vy(int n, double *alpha, double *beta, double *v1, double *v2, double *y)
{
    // intializes entire vectors v1, v2, x, and y
    // initialize constants alpha and beta
    double fn = (double)n;
    *alpha = 1.5;
    *beta = 1.2;

    for (int i = 0; i < n; i++)
    {

        v1[i] = ((i + 1) / fn) / 4.0;
        v2[i] = ((i + 1) / fn) / 6.0;
        y[i] = ((i + 1) / fn) / 8.0;
    }
}

void init_gemver_Au(int n, int m, int n_min, int n_max, double *A, double *u1, double *u2)
{
    // intializes part of array A from row n_min to n_max and all columns
    // initializes n elements of u1 and u2 and w from n_min to n_max

    double fn = (double)n;

    for (int i = 0; i < n_max - n_min; i++)
    {
        int i_value = i + n_min;
        u1[i] = i_value;
        u2[i] = ((i_value + 1) / fn) / 2.0;

        for (int j = 0; j < m; j++)
            A[i * m + j] = (double)(i_value * j % n) / n;
    }
}
void init_gemver_w(int n, double *w)
{
    // initializes  n elements of w

    for (int i = 0; i < n; i++)
    {
        w[i] = 0.0;
    }
}
void init_gemver_xz(int n, int n_min, int n_max, double *x, double *z)
{

    // initializes n elements of z from n_min to n_max

    double fn = (double)n;

    for (int i = 0; i < n_max - n_min; i++)
    {
        int i_value = i + n_min;
        z[i] = ((i_value + 1) / fn) / 9.0;
        x[i] = 0.0;
    }
}

void init_gemver_uy(int n ,  double *alpha, double *beta, double *u1, double *u2, double *y){
    // intializes entire vectors u1, u2, and y
    // initialize constants alpha and beta
    double fn = (double)n;
    *alpha = 1.5;
    *beta = 1.2;

    for (int i = 0; i < n; i++)
    {

        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        y[i] = ((i + 1) / fn) / 8.0;
    }   
}

void init_gemver_Av(int n, int m, int m_min, int m_max, double *A, double *v1, double *v2)
{
    // intializes part of array A from col m_min to m_max and all rows
    // initializes m_max-min elements of v1 and v2 from col m_min to m_max
    //A[i * m + j] = (double)(i * j % n) / n;
    // v1[j] = ((j + 1) / fn) / 4.0;
    // v2[j] = ((j + 1) / fn) / 6.0;

    double fn = (double)n;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m_max-m_min; j++){
            
            A[i * (m_max -m_min) + j] = (double)(i * (j+m_min) % n) / n;
        }
    }
    for (int j = 0; j < m_max-m_min; j++) {
        v1[j] = (((j+m_min) + 1) / fn) / 4.0;
        v2[j] = (((j+m_min) + 1) / fn) / 6.0;
    }
}