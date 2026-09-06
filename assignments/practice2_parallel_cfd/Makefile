CC=gcc
MPICC=mpicc
CFLAGS=-O3 -std=c11 -march=native
OMPFLAGS=-fopenmp
BIN=bin
SRC=src

.PHONY: all cpu clean

all: cpu

cpu: $(BIN)/stencil_serial $(BIN)/stencil_openmp $(BIN)/stencil_mpi $(BIN)/stencil_hybrid

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

clean:
	rm -rf $(BIN)
