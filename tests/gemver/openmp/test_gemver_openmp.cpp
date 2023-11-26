#include <gtest/gtest.h>
#include "../../../src/gemver/gemver_baseline.h"
#include "../../../src/helpers/gemver_init.h"
#include "../../../src/gemver/openmp/gemver_openmp.h"

TEST(gemverTest, SparseInitialization){
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

    sparse_init_gemver(n, &alpha, &beta, A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
    sparse_init_gemver(n, &alpha, &beta, A_baseline.data(), u1_baseline.data(), v1_baseline.data(), u2_baseline.data(), v2_baseline.data(), w_baseline.data(), x_baseline.data(), y_baseline.data(), z_baseline.data());

    gemver_openmp_v1(n, alpha, beta,A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
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


    rand_init_gemver(n, &alpha, &beta, A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
    rand_init_gemver(n, &alpha, &beta, A_baseline.data(), u1_baseline.data(), v1_baseline.data(), u2_baseline.data(), v2_baseline.data(), w_baseline.data(), x_baseline.data(), y_baseline.data(), z_baseline.data());

    //kernel_gemver represents the baseline
    gemver_openmp_v1(n, alpha, beta,A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
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
        init_gemver(n, &alpha, &beta, A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
        init_gemver(n, &alpha, &beta, A_baseline.data(), u1_baseline.data(), v1_baseline.data(), u2_baseline.data(), v2_baseline.data(), w_baseline.data(), x_baseline.data(), y_baseline.data(), z_baseline.data());

        gemver_openmp_v1(n, alpha, beta, A.data(), u1.data(), v1.data(), u2.data(), v2.data(), w.data(), x.data(), y.data(), z.data());
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