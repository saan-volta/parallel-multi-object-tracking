CXX      := g++
NVCC     := nvcc

# Compiler Flags
CXXFLAGS :=  -std=c++17 -Iinclude
NVCCFLAGS:=  -std=c++17 -Iinclude

SRC_DIR  := src
BUILD_DIR:= build
TARGET   := parallel_multi_obj_tracker

# Source and Object Files
CPP_SRCS := $(SRC_DIR)/main.cpp
CUDA_SRCS:= $(SRC_DIR)/kalman_filter.cu

CPP_OBJS := $(BUILD_DIR)/main.o
CUDA_OBJS:= $(BUILD_DIR)/kalman_filter.o
OBJS     := $(CPP_OBJS) $(CUDA_OBJS)

# Default Rule
all: $(TARGET)

# Link Object Files into Executable (Using NVCC links CUDA runtime automatically)
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@

# Compile C++ Source Files
$(CPP_OBJS): $(CPP_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA Source Files
$(CUDA_OBJS): $(CUDA_SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Create Build Directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean Build Artifacts
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
