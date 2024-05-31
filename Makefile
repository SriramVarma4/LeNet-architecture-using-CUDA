# Define compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11
NVCCFLAGS = -lcudart -arch=sm_35

# Define source files
CPP_FILES := $(wildcard src/*.cpp)
CU_FILES := $(wildcard src/*.cu)

# Define target names
TARGETS := subtask1 subtask2 subtask3 subtask4

# Define rules
.PHONY: all clean run_python_script

all: $(TARGETS)

subtask1: src/assignment2_subtask1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

subtask2: src/assignment2_subtask2.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

subtask3: src/assignment2_subtask3.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

subtask4: src/assignment2_subtask4.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run_python_script:
	python preprocessing.py

clean:
	rm -f $(TARGETS)
