#!/bin/bash
set -euo pipefail

if ! command -v mpicc >/dev/null 2>&1; then
    module load foss/2023b
fi

make mpi

if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun -n 4 ./bin/stencil_mpi --demo
else
    mpirun -n 4 ./bin/stencil_mpi --demo
fi
