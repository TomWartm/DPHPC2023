#include <gtest/gtest.h>
#include "../../../src/trisolv/trisolv_baseline.h"
#include "../../../src/trisolv/openmp/trisolv_openmp.h"
#include "../../../src/helpers/trisolv_init.h"

TEST(trisolvTest, IdentityInitialization){
    int n = 3;

    double *L = (double*) malloc((n * n) * sizeof(double));
    double *L_baseline = (double*) malloc((n * n) * sizeof(double));
    double *x = (double*) malloc((n) * sizeof(double));
    double *x_baseline = (double*) malloc((n) * sizeof(double));
    double *b = (double*) malloc((n) * sizeof(double));
    double *b_baseline = (double*) malloc((n) * sizeof(double));

    identity_trisolv(n, L, x, b);
    identity_trisolv(n, L_baseline, x_baseline, b_baseline);

    trisolv_openmp(n, L, x, b);
    trisolv_baseline(n, L_baseline, x_baseline, b_baseline);

    for (int j = 0; j < n * n; j++) {
        EXPECT_NEAR(L[j], L_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(x[j], x_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(b[j], b_baseline[j], 1e-6);
    }
    free((void*)L); free((void*)L_baseline);
    free((void*)x); free((void*)x_baseline);
    free((void*)b); free((void*)b_baseline);
}



TEST(trisolvTest, INIT_TRISOLV){
    int n = 3;

    double *L = (double*) malloc((n * n) * sizeof(double));
    double *L_baseline = (double*) malloc((n * n) * sizeof(double));
    double *x = (double*) malloc((n) * sizeof(double));
    double *x_baseline = (double*) malloc((n) * sizeof(double));
    double *b = (double*) malloc((n) * sizeof(double));
    double *b_baseline = (double*) malloc((n) * sizeof(double));

    init_trisolv(n, L, x, b);
    init_trisolv(n, L_baseline, x_baseline, b_baseline);

    trisolv_openmp(n, L, x, b);
    trisolv_baseline(n, L_baseline, x_baseline, b_baseline);

    for (int j = 0; j < n * n; j++) {
        EXPECT_NEAR(L[j], L_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(x[j], x_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(b[j], b_baseline[j], 1e-6);
    }
    free((void*)L); free((void*)L_baseline);
    free((void*)x); free((void*)x_baseline);
    free((void*)b); free((void*)b_baseline);
}



TEST(trisolvTest, RandomInitialization){
    int n = 5;
    double *L = (double*) malloc((n * n) * sizeof(double)); double *L_baseline = (double*) malloc((n * n) * sizeof(double));
    double *x = (double*) malloc((n) * sizeof(double)); double *x_baseline = (double*) malloc((n) * sizeof(double));
    double *b = (double*) malloc((n) * sizeof(double)); double *b_baseline = (double*) malloc((n) * sizeof(double));


    random_trisolv(n, L, x, b);
    random_trisolv(n, L_baseline, x_baseline, b_baseline);

    trisolv_openmp(n, L, x, b);
    trisolv_baseline(n, L_baseline, x_baseline, b_baseline);

    for (int j = 0; j < n * n; j++) {
        EXPECT_NEAR(L[j], L_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(x[j], x_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(b[j], b_baseline[j], 1e-6);
    }

    free((void*)L); free((void*)L_baseline);
    free((void*)x); free((void*)x_baseline);
    free((void*)b); free((void*)b_baseline);
}

TEST(trisolvTest, LTriangularInitialization){
    int n = 100;
    double *L = (double*) malloc((n * n) * sizeof(double)); double *L_baseline = (double*) malloc((n * n) * sizeof(double));
    double *x = (double*) malloc((n) * sizeof(double)); double *x_baseline = (double*) malloc((n) * sizeof(double));
    double *b = (double*) malloc((n) * sizeof(double)); double *b_baseline = (double*) malloc((n) * sizeof(double));


    lowertriangular_trisolv(n, L, x, b);
    lowertriangular_trisolv(n, L_baseline, x_baseline, b_baseline);

    trisolv_openmp(n, L, x, b);
    trisolv_baseline(n, L_baseline, x_baseline, b_baseline);

    for (int j = 0; j < n * n; j++) {
        EXPECT_NEAR(L[j], L_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(x[j], x_baseline[j], 1e-6);
    }
    for (int j = 0; j < n; j++) {
        EXPECT_NEAR(b[j], b_baseline[j], 1e-6);
    }

    free((void*)L); free((void*)L_baseline);
    free((void*)x); free((void*)x_baseline);
    free((void*)b); free((void*)b_baseline);
}

TEST(trisolvTest, DifferentSizes){
    std::vector<int> n_vec = {0, 1, 10, 100};

    for (auto n : n_vec){
        double *L = (double*) malloc((n * n) * sizeof(double)); double *L_baseline = (double*) malloc((n * n) * sizeof(double));
        double *x = (double*) malloc((n) * sizeof(double)); double *x_baseline = (double*) malloc((n) * sizeof(double));
        double *b = (double*) malloc((n) * sizeof(double)); double *b_baseline = (double*) malloc((n) * sizeof(double));

        init_trisolv(n, L, x, b);
        init_trisolv(n, L_baseline, x_baseline, b_baseline);

        trisolv_openmp(n, L, x, b);
        trisolv_baseline(n, L_baseline, x_baseline, b_baseline);

        for (int j = 0; j < n * n; j++) {
            EXPECT_NEAR(L[j], L_baseline[j], 1e-6);
        }
        for (int j = 0; j < n; j++) {
            EXPECT_NEAR(x[j], x_baseline[j], 1e-6);
        }
        for (int j = 0; j < n; j++) {
            EXPECT_NEAR(b[j], b_baseline[j], 1e-6);
        }

        free((void*)L); free((void*)L_baseline);
        free((void*)x); free((void*)x_baseline);
        free((void*)b); free((void*)b_baseline);
    }

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}