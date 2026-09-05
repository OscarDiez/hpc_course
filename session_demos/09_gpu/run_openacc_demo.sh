#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=== SciTech Session 9 GPU demo ==="
echo "Node: $(hostname)"
echo

echo "GPU allocated by Slurm:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found. Did this job receive a GPU node?"
  exit 2
fi

echo
if ! command -v nvc >/dev/null 2>&1; then
  cat <<'MSG'
OpenACC compiler 'nvc' is not available in the current environment.
SciTech has indicated that the accelerated EESSI module tree contains the GPU toolchain,
but that tree is not enabled by default.

Ask the instructor/SciTech for the exact module activation command, then rerun this script.
The Slurm GPU allocation itself is working correctly.
MSG
  exit 3
fi

echo "Compiler: $(command -v nvc)"
nvc --version | head -n 2 || true

echo
echo "--- Example 1: one OpenACC pragma, 1024 elements ---"
nvc -O2 -acc -Minfo=accel openacc_vector_add.c -o openacc_vector_add
./openacc_vector_add

echo
echo "--- Example 2: CPU vs GPU and the cost of data movement ---"
nvc -O3 -acc -Minfo=accel openacc_cpu_gpu_benchmark.c -o openacc_benchmark
./openacc_benchmark
