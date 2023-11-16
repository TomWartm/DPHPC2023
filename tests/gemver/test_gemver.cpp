#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../src/gemver/gemver_baseline.h"
#include <stdlib.h>
#include <time.h>

static void rand_init_array(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z) {

    srand(42);

    // Scaling factors
    double alpha_scale = 20.0, alpha_offset = -10.0;
    double beta_scale = 20.0, beta_offset = -10.0;

    // Matrix and vector values
    double A_scale = 2.0, A_offset = -1.0;

    *alpha = (double)rand() / RAND_MAX * alpha_scale + alpha_offset;
    *beta = (double)rand() / RAND_MAX * beta_scale + beta_offset;

    for (int i = 0; i < n; i++) {
        u1[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        u2[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        v1[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        v2[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        y[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        z[i] = (double)rand() / RAND_MAX * A_scale + A_offset;
        x[i] = 0.0; // Start with zero
        w[i] = 0.0; // Start with zero
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (double)rand() / RAND_MAX * A_scale + A_offset;
        }
    }
}

static void init_array(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z) {

    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double)n;

    for (int i = 0; i < n; i++)
    {
        u1[i] = i;
        u2[i] = ((i + 1.0) / fn) / 2.0;
        v1[i] = ((i + 1.0) / fn) / 4.0;
        v2[i] = ((i + 1.0) / fn) / 6.0;
        y[i] = ((i + 1.0) / fn) / 8.0;
        z[i] = ((i + 1.0) / fn) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (int j = 0; j < n; j++)
            A[i * n + j] = (double)(i * j % n) / n;
    }
}

TEST(gemverTest, RandomInitialization){

    int n = 10;
    double alpha;
    double beta;
    std::vector<double> A(n * n), A_baseline(n * n);
    std::vector<double> u1(n), u1_baseline(n);
    std::vector<double> v1(n), v1_baseline(n);
    std::vector<double> u2(n), u2_baseline(n);
    std::vector<double> v2(n), v2_baseline(n);
    std::vector<double> w(n), w_baseline(n);
    std::vector<double> x(n), x_baseline(n);
    std::vector<double> y(n), y_baseline(n);
    std::vector<double> z(n), z_baseline(n);


    rand_init_array(n, &alpha, &beta, A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
    rand_init_array(n, &alpha, &beta, A_baseline.data(), u1_baseline.data(), v1_baseline.data(), u2_baseline.data(), v2_baseline.data(), w_baseline.data(), x_baseline.data(), y_baseline.data(), z_baseline.data());

    //kernel_gemver represents the baseline
    kernel_gemver(n, alpha, beta,A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
    kernel_gemver(n, alpha, beta, A_baseline.data(), u1_baseline.data(), v1_baseline.data(), u2_baseline.data(), v2_baseline.data(), w_baseline.data(), x_baseline.data(), y_baseline.data(), z_baseline.data());

    for (int j = 0; j < n * n; j++) {
        EXPECT_NEAR(A[j], A_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(x[j], x_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(w[j], w_baseline[j], 1e-6);
    }

}

TEST(gemverTest, DifferentSizes){
    
    std::vector<int>  n_vec(5);
    n_vec = {0, 10, 100, 1000, 10000};
    double alpha;
    double beta;

    for (auto n : n_vec){
        std::vector<double> A(n * n), A_baseline(n * n);
        std::vector<double> u1(n), u1_baseline(n);
        std::vector<double> v1(n), v1_baseline(n);
        std::vector<double> u2(n), u2_baseline(n);
        std::vector<double> v2(n), v2_baseline(n);
        std::vector<double> w(n), w_baseline(n);
        std::vector<double> x(n), x_baseline(n);
        std::vector<double> y(n), y_baseline(n);
        std::vector<double> z(n), z_baseline(n);
        init_array(n, &alpha, &beta, A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
        init_array(n, &alpha, &beta, A_baseline.data(), u1_baseline.data(), v1_baseline.data(), u2_baseline.data(), v2_baseline.data(), w_baseline.data(), x_baseline.data(), y_baseline.data(), z_baseline.data());

        kernel_gemver(n, alpha, beta, A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
        kernel_gemver(n, alpha, beta, A_baseline.data(), u1_baseline.data(), v1_baseline.data(), u2_baseline.data(), v2_baseline.data(), w_baseline.data(), x_baseline.data(), y_baseline.data(), z_baseline.data());

        for (int j = 0; j < n * n; j++) {
        EXPECT_NEAR(A[j], A_baseline[j], 1e-6);
        }
        for (int j = 0; j < n; j++) {
        EXPECT_NEAR(x[j], x_baseline[j], 1e-6);
        }
        for (int j = 0; j < n; j++) {
        EXPECT_NEAR(w[j], w_baseline[j], 1e-6);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}