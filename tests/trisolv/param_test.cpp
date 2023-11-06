#include <gtest/gtest.h>
#include <stdlib.h>
#include <iostream>

#include "trisolv.h"
#include "helper.h"

class TrisolvTest :
	public testing::TestWithParam<std::vector<TestCase>> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.

};

TEST_P(TrisolvTest, checkResult) {
	TestCase test = GetParam();
	std::cout << "Test cases: " << test.test_name << "\n";
	double* A = (double*)malloc(test.N * test.N * sizeof(double));
	double* x = (double*)malloc(test.N * sizeof(double));
	double* b = (double*)malloc(test.N * sizeof(double));
	init_matrix(A, test.A, test.N);
	init_vector(x, test.x, test.N);
	init_vector(b, test.b, test.N);
	kernel_trisolv(test.N, A, x, b);
	EXPECT_TRUE(check_result(x, test.x, test.N, test.Epsilon));
	free(A);
	free(x);
	free(b);
}

INSTANTIATE_TEST_SUITE_P(TrisolvTest,
                         checkResult,
                         testing::Values(test_cases));


int main(int argc, char** argv) {
	const std::vector<TestCase> test_cases = parse_tests("test_matrix.txt");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
