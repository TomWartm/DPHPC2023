#include <gtest/gtest.h>

#include "../src/trisolve/trisolve_baseline.h"
#include "../src/trisolve/trisolve_init.h"


TEST(trisolveTest, kernel_trisolve){
    int n = 2000;
    double *L = (double *)malloc((n * n) * sizeof(double));
    double *x = (double *)malloc((n) * sizeof(double));
    double *b = (double *)malloc((n) * sizeof(double));

    init_trisolve(n, *L, *x, *b);
    kernel_trisolve(n, *L, *x, *b);

    ASSERT_EQ(1,1); // TODO: check results

    free((void*)L);
    free((void*)x);
    free((void*)b);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}