CC=gcc
MPICC=mpicc
NVCC=nvcc
ACC_CC?=gcc
CFLAGS=-O3 -std=gnu11 -march=native
OMPFLAGS=-fopenmp
ACCFLAGS?=-fopenacc
BIN=bin
SRC=src

.PHONY: all serial openmp mpi hybrid openacc cuda clean

all: serial

serial: $(BIN)/stencil_serial
openmp: $(BIN)/stencil_openmp
mpi: $(BIN)/stencil_mpi
hybrid: $(BIN)/stencil_hybrid
openacc: $(BIN)/stencil_openacc
cuda: $(BIN)/stencil_cuda

$(BIN):
	mkdir -p $(BIN)

$(BIN)/stencil_serial: $(SRC)/stencil_serial.c | $(BIN)
	$(CC) $(CFLAGS) $< -lm -o $@

$(BIN)/stencil_openmp: $(SRC)/stencil_openmp_student.c | $(BIN)
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -lm -o $@

$(BIN)/stencil_mpi: $(SRC)/stencil_mpi_student.c | $(BIN)
	@command -v $(MPICC) >/dev/null 2>&1 || { echo "ERROR: mpicc is not available. Load the SciTech MPI environment first."; exit 1; }
	$(MPICC) $(CFLAGS) $< -lm -o $@

$(BIN)/stencil_hybrid: $(SRC)/stencil_hybrid.c | $(BIN)
	@command -v $(MPICC) >/dev/null 2>&1 || { echo "ERROR: mpicc is not available. Load the SciTech MPI environment first."; exit 1; }
	$(MPICC) $(CFLAGS) $(OMPFLAGS) $< -lm -o $@

$(BIN)/stencil_openacc: $(SRC)/stencil_openacc_student.c | $(BIN)
	@command -v $(ACC_CC) >/dev/null 2>&1 || { echo "ERROR: OpenACC compiler '$(ACC_CC)' is not available in the current environment."; exit 1; }
	$(ACC_CC) $(CFLAGS) $(ACCFLAGS) $< -lm -o $@

$(BIN)/stencil_cuda: $(SRC)/stencil_cuda_student.cu | $(BIN)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "ERROR: nvcc is not available in the current environment. Load the SciTech CUDA/EESSI accelerator environment first."; exit 1; }
	$(NVCC) -O3 $< -o $@

clean:
	rm -rf $(BIN)
