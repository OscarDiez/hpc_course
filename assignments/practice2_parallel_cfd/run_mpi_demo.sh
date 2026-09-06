#!/bin/bash
set -euo pipefail

module purge
module load foss/2023b

make mpi

mpirun -n 4 \
  --mca pml ob1 --mca btl self,vader,tcp \
  ./bin/stencil_mpi --demo
