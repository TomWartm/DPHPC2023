#include <gtest/gtest.h>
#include <iostream>

#include "../../src/gemver/gemver_baseline.h"
#include "../../src/helpers/gemver_init.h"


TEST(gemverTest, kernel_gemver){

    int n = 10;
    double alpha;
    double beta;
    double *A = (double *)malloc((n * n) * sizeof(double));
    double *A_baseline = (double *)malloc((n * n) * sizeof(double));
    double *u1 = (double *)malloc((n) * sizeof(double));
    double *u1_baseline = (double *)malloc((n) * sizeof(double));
    double *v1 = (double *)malloc((n) * sizeof(double));
    double *v1_baseline = (double *)malloc((n) * sizeof(double));
    double *u2 = (double *)malloc((n) * sizeof(double));
    double *u2_baseline = (double *)malloc((n) * sizeof(double));
    double *v2 = (double *)malloc((n) * sizeof(double));
    double *v2_baseline = (double *)malloc((n) * sizeof(double));
    double *w = (double *)malloc((n) * sizeof(double));
    double *w_baseline= (double *)malloc((n) * sizeof(double));
    double *x = (double *)malloc((n) * sizeof(double));
    double *x_baseline= (double *)malloc((n) * sizeof(double));
    double *y = (double *)malloc((n) * sizeof(double));
    double *y_baseline= (double *)malloc((n) * sizeof(double));
    double *z = (double *)malloc((n) * sizeof(double));
    double *z_baseline= (double *)malloc((n) * sizeof(double));

    init_gemver(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
    init_gemver(n, &alpha, &beta, A_baseline, u1_baseline, v1_baseline, u2_baseline, v2_baseline, w_baseline, x_baseline, y_baseline, z_baseline);

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
