#include <gtest/gtest.h>

#include "../../src/trisolv/trisolv_baseline.h"
#include "../../src/helpers/trisolv_init.h"
#include "helper.h"


TEST(trisolveTest, kernel_trisolv){
    int n = 2000;
    double *L = (double*) malloc((n * n) * sizeof(double));
    double *x = (double*) malloc((n) * sizeof(double));
    double *b = (double*) malloc((n) * sizeof(double));

    init_trisolv(n, L, x, b);
    kernel_trisolv(n, L, x, b);

    ASSERT_EQ(1,1); // TODO: check results

    free((void*)L);
    free((void*)x);
    free((void*)b);
}

TEST(trisolvTest, from_file) {
    std::vector<TestCase> test_cases = parse_tests("tests/trisolv/test_matrix.txt");
    std::cout << "Test cases:\n";
    for (const TestCase& t: test_cases) std::cout << t.test_name << "\n";

    std::cout << "\n\n";
    for (TestCase& test : test_cases) {
        double* L = (double*)malloc(test.N * test.N * sizeof(double));
        double* x = (double*)malloc(test.N * sizeof(double));
        double* b = (double*)malloc(test.N * sizeof(double));
        init_matrix(L, test.A, test.N);
        init_vector(x, test.x, test.N);
        init_vector(b, test.b, test.N);
        kernel_trisolv(test.N, L, x, b);
        bool correct = check_result(x, test.x, test.N, test.Epsilon);
        EXPECT_TRUE(correct);
        if (!correct) std::cout << "FAILED " << test.test_name << "\n";
        free(L);
        free(x);
        free(b);
    }
}