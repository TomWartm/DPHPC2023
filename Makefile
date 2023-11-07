# Compiler flags
CXXFLAGS = -std=c++14 -Wall

# MPI flags
MPIFLAGS = -I/path/to/mpi/include -L/path/to/mpi/lib -lmpi

# Source file directories
SRC_DIR = src
GEMVER_DIR = $(SRC_DIR)/gemver
HELPERS_DIR = $(SRC_DIR)/helpers
HELPERS_MPI_DIR = $(SRC_DIR)/helpers/mpi
GEMVER_MPI_DIR = $(SRC_DIR)/gemver/mpi



## Timing
# Compile and run evaluate_gemver.cpp
GEMVER_EXECUTALE = my_evaluate_gemver
GEMVER_EXECUTABLE_SRC = src/evaluate_gemver.cpp

$(GEMVER_EXECUTALE): $(GEMVER_EXECUTABLE_SRC)  $(wildcard $(GEMVER_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp)
	g++ $(CXXFLAGS)  -o $@ $^

gemver: $(GEMVER_EXECUTALE)
	./$(GEMVER_EXECUTALE)  


# Compile and run evaluate_gemver_mpi.cpp
GEMVER_MPI_EXECUTALE = my_evaluate_gemver_mpi
GEMVER_MPI_EXECUTABLE_SRC = src/evaluate_gemver_mpi.cpp

$(GEMVER_MPI_EXECUTALE): $(GEMVER_MPI_EXECUTABLE_SRC) $(wildcard $(GEMVER_DIR)/*.cpp) $(wildcard $(GEMVER_MPI_DIR)/*.cpp) $(wildcard $(HELPERS_MPI_DIR)/*.cpp)
	mpicxx $(CXXFLAGS) -o $@ $^ $(MPIFLAGS) 

gemver_mpi: $(GEMVER_MPI_EXECUTALE)
	mpirun -np 3 ./$(GEMVER_MPI_EXECUTALE)  
	



## Testing
# Compile and run test_gemver.cpp
TEST_GEMVER_EXECUTABLE = my_test_gemver
TEST_GEMVER_DIR = tests/gemver

$(TEST_GEMVER_EXECUTABLE): $(wildcard $(TEST_GEMVER_DIR)/*.cpp) $(wildcard $(GEMVER_DIR)/*.cpp)
	g++ $(CXXFLAGS)  -o $@ $^ -lgtest -lgtest_main

test_gemver: $(TEST_GEMVER_EXECUTABLE)
	./$(TEST_GEMVER_EXECUTABLE)  

# Compile and run test_gemver_mpi.cpp
TEST_GEMVER_MPI_EXECUTABLE = my_test_gemver_mpi
TEST_GEMVER_MPI_DIR = tests/gemver/mpi

$(TEST_GEMVER_MPI_EXECUTABLE): $(wildcard $(TEST_GEMVER_MPI_DIR)/*.cpp) $(wildcard $(GEMVER_DIR)/*.cpp) $(wildcard $(GEMVER_MPI_DIR)/*.cpp)
	mpicxx $(CXXFLAGS) -o $@ $^ $(MPIFLAGS) -lgtest -lgtest_main

test_gemver_mpi: $(TEST_GEMVER_MPI_EXECUTABLE)
	mpirun -np 2 ./$(TEST_GEMVER_MPI_EXECUTABLE)  


# remove all .o and executable files
clean: 
	rm -f *.o $(GEMVER_EXECUTALE) $(GEMVER_MPI_EXECUTALE) $(TEST_GEMVER_EXECUTABLE) $(TEST_GEMVER_MPI_EXECUTABLE)