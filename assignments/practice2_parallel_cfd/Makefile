CC=gcc
MPICC=mpicc
NVCC=nvcc
CFLAGS=-O3 -std=gnu11 -march=native
OMPFLAGS=-fopenmp
BIN=bin
SRC=src

.PHONY: all cpu serial openmp mpi hybrid cuda clean

all: cpu

# Build all CPU implementations that are currently provided.
cpu: serial openmp mpi hybrid

# Individual targets used by the student instructions.
serial: $(BIN)/stencil_serial
openmp: $(BIN)/stencil_openmp
mpi: $(BIN)/stencil_mpi
hybrid: $(BIN)/stencil_hybrid
cuda: $(BIN)/stencil_gpu

$(BIN):
	mkdir -p $(BIN)

$(BIN)/stencil_serial: $(SRC)/stencil_serial.c | $(BIN)
	$(CC) $(CFLAGS) $< -lm -o $@

$(BIN)/stencil_openmp: $(SRC)/stencil_openmp.c | $(BIN)
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -lm -o $@

$(BIN)/stencil_mpi: $(SRC)/stencil_mpi.c | $(BIN)
	$(MPICC) $(CFLAGS) $< -lm -o $@

$(BIN)/stencil_hybrid: $(SRC)/stencil_hybrid.c | $(BIN)
	$(MPICC) $(CFLAGS) $(OMPFLAGS) $< -lm -o $@

$(BIN)/stencil_gpu: $(SRC)/stencil_gpu.cu | $(BIN)
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "ERROR: nvcc is not available in the current environment. Load the SciTech CUDA/EESSI accelerator environment first."; exit 1; }
	$(NVCC) -O3 $< -o $@

clean:
	rm -rf $(BIN)
