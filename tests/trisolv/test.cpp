#include <gtest/gtest.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>

#include "trisolv.h"

struct TestCase {
	std::string test_name;
	double Epsilon = 0.0009765625;
	int N;
	std::vector<std::vector<double>> A;
	std::vector<double> x;
	std::vector<double> b;
};

std::vector<TestCase> parse_tests(const std::string& filename) {
	std::ifstream file(filename);
	std::vector<TestCase> cases;
	if (!file.is_open())
        std::cout << "failed to open " << filename << '\n';
    else {
		std::string line, elem;
		while (std::getline(file, line)) {
			while (line.size() == 1) std::getline(file, line); //discard empty lines
			TestCase test_case;
			//parse test_name
			test_case.test_name = line;
			//std::cout << "[NAME]" << test_case.test_name << "\n";
			std::getline(file, line);
			//parse Epsilon if defined in file
			if (line.find('E') != line.npos) {
				line.pop_back();
				test_case.Epsilon = std::stod(line);
				//std::cout << "[EPSILON]" << test_case.Epsilon << "\n";
				std::getline(file, line);
			}
			//parse N
			test_case.N = std::stoi(line);
			//std::cout << "[N]" << test_case.N << "\n";
			std::getline(file, line);
			//parse A
			for (int i = 0; i < test_case.N; ++i) {
				//std::cout << "[A]";
				std::stringstream stream(line);
				std::vector<double> row;
				for (int j = 0; j < test_case.N; ++j) {
					stream >> elem;
					row.push_back(std::stod(elem));
					//std::cout << row.back() << " ";
				}
				test_case.A.push_back(row);
				//std::cout << "\n";
				std::getline(file, line);
			}
			//parse x
			{
				//std::cout << "[x]";
				std::stringstream stream(line);
				std::vector<double> row;
				for (int j = 0; j < test_case.N; ++j) {
					stream >> elem;
					row.push_back(std::stod(elem));
					//std::cout << row.back() << " ";
				}
				test_case.x = row;
				//std::cout << "\n";
				std::getline(file, line);
			}
			//parse b
			{
				//std::cout << "[b]";
				std::stringstream stream(line);
				std::vector<double> row;
				for (int j = 0; j < test_case.N; ++j) {
					stream >> elem;
					row.push_back(std::stod(elem));
					//std::cout << row.back() << " ";
				}
				test_case.b = row;
				//std::cout << "\n";
				std::getline(file, line);
			}
			cases.push_back(test_case);
		}
		//std::cout << "size = " << cases.size() << "\n";
    }
	return cases;
}

void init_matrix(double* T, const std::vector<std::vector<double>>& S, int N) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			T[i * N + j] = S[i][j];
		}
	}
}

void init_vector(double* T, const std::vector<double>& S, int N) {
	for (int j = 0; j < N; ++j) {
		T[j] = S[j];
	}
}

bool check_result(double* T, const std::vector<double>& S, int N, double Epsilon) {
	for (int j = 0; j < N; ++j) {
		if (std::abs(T[j] - S[j]) > Epsilon) return false;
	}
	return true;
}

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