# Makefile for FEM Assembly Pipeline
# Compiles mesh parser, CSR builder, CUDA kernels, and main program

# Compilers
NVCC = nvcc
CXX = g++

# Directories
CPU_DIR = CPU
GPU_DIR = GPU

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_75 -std=c++14
CXX_FLAGS = -O3 -std=c++14 -Wall

# Include directories
INCLUDES = -I$(CPU_DIR) -I$(GPU_DIR)

# Source files
CPU_SOURCES = $(CPU_DIR)/mesh_parser.cpp $(CPU_DIR)/csr_builder.cpp $(CPU_DIR)/reorder.cpp
GPU_SOURCES = $(GPU_DIR)/localSolve.cu $(GPU_DIR)/globalAsm.cu $(GPU_DIR)/gpu_solve_csr.cu
MAIN_SOURCE = main.cpp

# Object files
CPU_OBJECTS = $(CPU_SOURCES:.cpp=.o)
GPU_OBJECTS = $(GPU_SOURCES:.cu=.o)
MAIN_OBJECT = $(MAIN_SOURCE:.cpp=.o)

# Executable
TARGET = fem_assembly

# Default target
all: $(TARGET)

# Link everything together
$(TARGET): $(CPU_OBJECTS) $(GPU_OBJECTS) $(MAIN_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^ -lcudart -lcusparse -lcublas -lnvToolsExt
	rm -f $(CPU_OBJECTS) $(GPU_OBJECTS) $(MAIN_OBJECT)

# Compile CPU source files
$(CPU_DIR)/%.o: $(CPU_DIR)/%.cpp $(CPU_DIR)/%.h
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile GPU source files
$(GPU_DIR)/%.o: $(GPU_DIR)/%.cu $(GPU_DIR)/%.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile main (needs CUDA for linking)
$(MAIN_OBJECT): $(MAIN_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET) $(CPU_OBJECTS) $(GPU_OBJECTS) $(MAIN_OBJECT)

tidy:
	rm -f $(CPU_OBJECTS) $(GPU_OBJECTS) $(MAIN_OBJECT)

profile-clean:
	rm -rf large_profile_results/
	rm -rf xlarge_profile_results/

out-clean:
	rm -r *.out
	rm -r *.err

# Phony targets
.PHONY: all clean

