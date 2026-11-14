#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bin lib/v1 lib/v2 lib/fixed
stage=${1:-cpu}
# CC can select the C compiler; keep it consistent with the MPI stack.
cc=${CC:-gcc}
# Build the same mathematics with different copy, optimisation and threading choices.
if [[ "$stage" == serial || "$stage" == cpu ]]; then
 "$cc" -std=c11 -O0 -g -Wall -Wextra src/heat_naive.c -o bin/heat_naive
 "$cc" -std=c11 -O0 -g -Wall -Wextra src/heat_cpu.c -o bin/heat_serial_O0
 "$cc" -std=c11 -O2 -g -Wall -Wextra src/heat_cpu.c -o bin/heat_serial
 "$cc" -std=c11 -O2 -g -Wall -Wextra -fopenmp src/heat_cpu.c -o bin/heat_omp
fi
# mpicc supplies MPI headers/libraries; -fopenmp adds local threads.
if [[ "$stage" == mpi || "$stage" == cpu ]]; then
 command -v mpicc
 mpicc -std=c11 -O2 -g -Wall -Wextra src/heat_mpi.c -o bin/heat_mpi
 mpicc -std=c11 -O2 -g -Wall -Wextra -fopenmp src/heat_mpi.c -o bin/heat_hybrid
fi
# Build interchangeable shared objects, then link one unchanged client executable.
if [[ "$stage" == libraries || "$stage" == cpu ]]; then
 for pair in 'v1 temperature_v1.c' 'v2 temperature_v2.c' 'fixed temperature_v2_fixed.c'; do
  read -r version source <<< "$pair"
  "$cc" -O2 -fPIC -shared "src/$source" -o "lib/$version/libtemperature.so"
 done
 "$cc" -std=c11 -O2 -g -DHOT_FROM_LIBRARY src/heat_cpu.c -Llib/v1 -ltemperature -o bin/heat_library
fi
# CUDA and OpenACC use different compilers but solve the same heat problem.
if [[ "$stage" == gpu ]]; then
 command -v nvcc
 nvcc -O2 -lineinfo src/heat_cuda.cu -o bin/heat_cuda
 if command -v nvc >/dev/null; then
  nvc -O2 -acc=gpu -Minfo=accel src/heat_acc.c -o bin/heat_acc 2>bin/openacc-build.txt
 else
  echo 'SKIPPED OpenACC: nvc is not available' >bin/openacc-build.txt
 fi
 "$cc" -std=c11 -O2 src/heat_cpu.c -o bin/heat_serial
fi
case "$stage" in serial|mpi|libraries|cpu|gpu) ;; *) echo "Unknown build stage: $stage" >&2; exit 2;; esac
