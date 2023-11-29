# Compiler flags
CXXFLAGS = -std=c++14 -Wall -O3 -ffast-math -fopenmp

# MPI flags
MPIFLAGS = -I/path/to/mpi/include -L/path/to/mpi/lib -lmpi

# Source file directories
SRC_DIR = src
HELPERS_DIR = $(SRC_DIR)/helpers
HELPERS_MPI_DIR = $(SRC_DIR)/helpers/mpi
GEMVER_DIR = $(SRC_DIR)/gemver
GEMVER_OPENMP_DIR = $(SRC_DIR)/gemver/openmp
GEMVER_MPI_DIR = $(SRC_DIR)/gemver/mpi
TRISOLV_DIR = $(SRC_DIR)/trisolv
TRISOLV_OPENMP_DIR = $(SRC_DIR)/trisolv/openmp
TRISOLV_MPI_DIR = $(SRC_DIR)/trisolv/mpi


## Timing
# Compile and run evaluate_gemver_openmp.cpp
GEMVER_OPENMP_EXECUTABLE = evaluate_gemver_openmp
GEMVER_OPENMP_EXECUTABLE_SRC = src/evaluate_gemver.cpp

$(GEMVER_OPENMP_EXECUTABLE): $(GEMVER_OPENMP_EXECUTABLE_SRC)  $(wildcard $(GEMVER_DIR)/*.cpp) $(wildcard $(GEMVER_OPENMP_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp)
	g++ $(CXXFLAGS)  -o $@ $^

gemver_openmp: $(GEMVER_OPENMP_EXECUTABLE)
	./$(GEMVER_OPENMP_EXECUTABLE)


# Compile and run evaluate_gemver_mpi.cpp
GEMVER_MPI_EXECUTABLE = evaluate_gemver_mpi
GEMVER_MPI_EXECUTABLE_SRC = src/evaluate_gemver_mpi.cpp

$(GEMVER_MPI_EXECUTABLE): $(GEMVER_MPI_EXECUTABLE_SRC) $(wildcard $(GEMVER_DIR)/*.cpp) $(wildcard $(GEMVER_MPI_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp) $(wildcard $(HELPERS_MPI_DIR)/*.cpp)
	mpicxx $(CXXFLAGS) -o $@ $^ $(MPIFLAGS)

gemver_mpi: $(GEMVER_MPI_EXECUTABLE)
	mpirun -np 3 ./$(GEMVER_MPI_EXECUTABLE)


# Compile and run evaluate_trisolv_openmp.cpp
TRISOLV_OPENMP_EXECUTABLE = evaluate_trisolv_openmp
TRISOLV_OPENMP_EXECUTABLE_SRC = src/evaluate_trisolv_openmp.cpp

$(TRISOLV_OPENMP_EXECUTABLE): $(TRISOLV_OPENMP_EXECUTABLE_SRC)  $(wildcard $(TRISOLV_DIR)/*.cpp) $(wildcard $(TRISOLV_OPENMP_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp)
	g++ $(CXXFLAGS)  -o $@ $^

trisolv: $(TRISOLV_EXECUTABLE)
	./$(TRISOLV_OPENMP_EXECUTABLE)


# Compile and run evaluate_trisolv_mpi.cpp
TRISOLV_MPI_EXECUTABLE = evaluate_trisolv_mpi
TRISOLV_MPI_EXECUTABLE_SRC = src/evaluate_trisolv_mpi.cpp

$(TRISOLV_MPI_EXECUTABLE): $(TRISOLV_MPI_EXECUTABLE_SRC) $(wildcard $(TRISOLV_DIR)/*.cpp) $(wildcard $(TRISOLV_MPI_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp) $(wildcard $(HELPERS_MPI_DIR)/*.cpp)
	mpicxx $(CXXFLAGS) -o $@ $^ $(MPIFLAGS)

trisolv_mpi: $(TRISOLV_MPI_EXECUTABLE)
	mpirun -np 4 ./$(TRISOLV_MPI_EXECUTABLE)


## Testing
# Compile and run test_gemver_openmp.cpp
TEST_GEMVER_OPENMP_EXECUTABLE = build_test_gemver_openmp
TEST_GEMVER_OPENMP_DIR = tests/gemver/openmp

$(TEST_GEMVER_OPENMP_EXECUTABLE): $(wildcard $(TEST_GEMVER_OPENMP_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp) $(wildcard $(GEMVER_DIR)/*.cpp) $(wildcard $(GEMVER_OPENMP_DIR)/*.cpp)
	g++ $(CXXFLAGS)  -o $@ $^ -lgtest -lgtest_main

test_gemver_openmp: $(TEST_GEMVER_OPENMP_EXECUTABLE)
	./$(TEST_GEMVER_OPENMP_EXECUTABLE)

# Compile and run test_gemver.cpp
TEST_GEMVER_EXECUTABLE = build_test_gemver
TEST_GEMVER_DIR = tests/gemver

$(TEST_GEMVER_EXECUTABLE): $(wildcard $(TEST_GEMVER_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp) $(wildcard $(GEMVER_DIR)/*.cpp) 
	g++ $(CXXFLAGS)  -o $@ $^ -lgtest -lgtest_main

test_gemver: $(TEST_GEMVER_EXECUTABLE)
	./$(TEST_GEMVER_EXECUTABLE)

# Compile and run test_gemver_mpi.cpp
TEST_GEMVER_MPI_EXECUTABLE = build_test_gemver_mpi
TEST_GEMVER_MPI_DIR = tests/gemver/mpi

$(TEST_GEMVER_MPI_EXECUTABLE): $(wildcard $(TEST_GEMVER_MPI_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp) $(wildcard $(GEMVER_DIR)/*.cpp) $(wildcard $(GEMVER_MPI_DIR)/*.cpp)
	mpicxx $(CXXFLAGS) -o $@ $^ $(MPIFLAGS) -lgtest -lgtest_main

test_gemver_mpi: $(TEST_GEMVER_MPI_EXECUTABLE)
	mpirun -np 4 ./$(TEST_GEMVER_MPI_EXECUTABLE)


# Compile and run test_trisolv_openmp.cpp
TEST_TRISOLV_OPENMP_EXECUTABLE = build_test_trisolv_openmp
TEST_TRISOLV_OPENMP_DIR = tests/trisolv/openmp

$(TEST_TRISOLV_OPENMP_EXECUTABLE): $(wildcard $(TEST_TRISOLV_OPENMP_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp) $(wildcard $(TRISOLV_DIR)/*.cpp) $(wildcard $(TRISOLV_OPENMP_DIR)/*.cpp)
	g++ $(CXXFLAGS)  -o $@ $^ -lgtest -lgtest_main

test_trisolv_openmp: $(TEST_TRISOLV_OPENMP_EXECUTABLE)
	./$(TEST_TRISOLV_OPENMP_EXECUTABLE)


# Compile and run test_trisolv_mpi.cpp
TEST_TRISOLV_MPI_EXECUTABLE = build_test_trisolv_mpi
TEST_TRISOLV_MPI_DIR = tests/trisolv/mpi

$(TEST_TRISOLV_MPI_EXECUTABLE): $(wildcard $(TEST_TRISOLV_MPI_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp) $(wildcard $(TRISOLV_DIR)/*.cpp) $(wildcard $(TRISOLV_MPI_DIR)/*.cpp)
	mpicxx $(CXXFLAGS) -o $@ $^ $(MPIFLAGS) -lgtest -lgtest_main

test_trisolv_mpi: $(TEST_TRISOLV_MPI_EXECUTABLE)
	mpirun -np 2 ./$(TEST_TRISOLV_MPI_EXECUTABLE)

# remove all .o and executable files
clean:
	rm -f *.o $(GEMVER_OPENMP_EXECUTABLE) $(GEMVER_MPI_EXECUTABLE) $(TRISOLV_OPENMP_EXECUTABLE) $(TRISOLV_MPI_EXECUTABLE) $(TEST_GEMVER_OPENMP_EXECUTABLE) $(TEST_GEMVER_MPI_EXECUTABLE) $(TEST_TRISOLV_OPENMP_EXECUTABLE) $(TEST_TRISOLV_MPI_EXECUTABLE)