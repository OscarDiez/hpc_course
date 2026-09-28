CC = gcc
CFLAGS ?= -O3 -std=gnu11 -march=native

BIN_DIR := bin
SRC := src/p3_pipeline_student.c

.PHONY: all clean

all: $(BIN_DIR)/p3_pipeline $(BIN_DIR)/p3_pipeline_gprof

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/p3_pipeline: $(SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC) -o $@

$(BIN_DIR)/p3_pipeline_gprof: $(SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) -pg $(SRC) -o $@

clean:
	rm -rf $(BIN_DIR) gmon.out *.bin p3_evidence.txt
