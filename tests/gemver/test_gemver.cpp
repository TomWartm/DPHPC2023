#include <gtest/gtest.h>
#include <iostream>

#include "../../src/gemver/gemver_baseline.h"


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
    double *A, *A_baseline = (double *)malloc((n * n) * sizeof(double));
    double *u1, *u1_baseline = (double *)malloc((n) * sizeof(double));
    double *v1, *v1_baseline = (double *)malloc((n) * sizeof(double));
    double *u2, *u2_baseline = (double *)malloc((n) * sizeof(double));
    double *v2, *v2_baseline = (double *)malloc((n) * sizeof(double));
    double *w , *w_baseline= (double *)malloc((n) * sizeof(double));
    double *x , *x_baseline= (double *)malloc((n) * sizeof(double));
    double *y , *y_baseline= (double *)malloc((n) * sizeof(double));
    double *z , *z_baseline= (double *)malloc((n) * sizeof(double));

    init_array(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
    init_array(n, &alpha, &beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);

    kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
    kernel_gemver(n, alpha, beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);

    for (int i = 0; i < n * n; i++) {
        ASSERT_NEAR(A[i], A_baseline[i], 1e-6);
    }
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(x[i], x_baseline[i], 1e-6);
    }
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(w[i], w_baseline[i], 1e-6);
    }
    //ASSERT_EQ(1,1); // TODO: check results

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

    free((void *)A_baseline);
    free((void *)u1_baseline);
    free((void *)v1_baseline);
    free((void *)u2_baseline);
    free((void *)v2_baseline);
    free((void *)w_baseline);
    free((void *)x_baseline);
    free((void *)y_baseline);
    free((void *)z_baseline);
}








int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
