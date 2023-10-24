#include <gtest/gtest.h>
#include <iostream>

#include "../src/gemver/gemver_baseline.h"


static void init_array(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z) {

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

TEST(gemverTest, kernel_gemver){
    
    int n = 10;
    double alpha;
    double beta;
    double *A = (double *)malloc((n * n) * sizeof(double));
    double *u1 = (double *)malloc((n) * sizeof(double));
    double *v1 = (double *)malloc((n) * sizeof(double));
    double *u2 = (double *)malloc((n) * sizeof(double));
    double *v2 = (double *)malloc((n) * sizeof(double));
    double *w = (double *)malloc((n) * sizeof(double));
    double *x = (double *)malloc((n) * sizeof(double));
    double *y = (double *)malloc((n) * sizeof(double));
    double *z = (double *)malloc((n) * sizeof(double));

    init_array(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);

    kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);

    ASSERT_EQ(1,1); // TODO: check results

    // free memory
    free((void *)A);
    free((void *)u1);
    free((void *)v1);
    free((void *)u2);
    free((void *)v2);
    free((void *)w);
    free((void *)x);
    free((void *)y);
    free((void *)z);
}








int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}