#!/bin/bash -l
# Read-only environment inspection; no jobs or packages are installed here.
echo "HOST=$(hostname)"
module -t list 2>&1 || true
type -a gcc mpicc python3 apptainer 2>&1 || true
gcc --version 2>&1 | head -1 || true
mpicc --showme 2>&1 || true
apptainer --version 2>&1 || true
# Compare compiler/MPI wrapper paths; do not mix system MPI and another stack.
