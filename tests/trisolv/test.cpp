#include <gtest/gtest.h>
#include <stdlib.h>
#include <iostream>

#include "trisolv.h"
#include "helper.h"



TEST(trisolvTest, from_file) {
	std::vector<TestCase> test_cases = parse_tests("test_matrix.txt");
	std::cout << "Test cases:\n";
	for (const TestCase& t: test_cases) std::cout << t.test_name << "\n";
	std::cout << "\n\n";
	for (int i = 0; i < test_cases.size(); ++i) {
		const TestCase& test = test_cases[i];
		double* A = (double*)malloc(test.N * test.N * sizeof(double));
		double* x = (double*)malloc(test.N * sizeof(double));
		double* b = (double*)malloc(test.N * sizeof(double));
		init_matrix(A, test.A, test.N);
		init_vector(x, test.x, test.N);
		init_vector(b, test.b, test.N);
		kernel_trisolv(test.N, A, x, b);
		bool correct = check_result(x, test.x, test.N, test.Epsilon);
		EXPECT_TRUE(correct);
		if (!correct) std::cout << "FAILED " << test.test_name << "\n";
		free(A);
		free(x);
		free(b);
	}
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
