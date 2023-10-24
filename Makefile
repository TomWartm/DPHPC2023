# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall

# Source file directories
SRC_DIR = src
GEMVER_DIR = $(SRC_DIR)/gemver
HELPERS_DIR = $(SRC_DIR)/helpers

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(GEMVER_DIR)/*.cpp) $(wildcard $(HELPERS_DIR)/*.cpp)

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXECUTABLE = my_program

# Build rule
$(EXECUTABLE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Run rule
run: $(EXECUTABLE)
	./$(EXECUTABLE)
