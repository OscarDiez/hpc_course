#!/bin/bash
#SBATCH --job-name=m3s4_lab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -eo pipefail
# Arguments: runner settings run-directory Python executable (captured from kernel).
runner=$1
settings=$2
run_dir=$3
kernel_python=$4
if ! command -v module >/dev/null 2>&1 && [ -r /cvmfs/software.eessi.io/versions/2023.06/init/lmod/bash ]; then
  source /cvmfs/software.eessi.io/versions/2023.06/init/lmod/bash
fi
if command -v module >/dev/null 2>&1; then
  module load foss/2023b
  # Optional packages: select a build compatible with the GCC/gompi 2023b toolchain.
  for package in FFTW HDF5; do
    candidate=$(module -t avail "$package" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | awk '/^(FFTW|HDF5)\// && /(13\.2\.0|2023b)/ {print $1; exit}')
    if [ -n "$candidate" ]; then module load "$candidate"; fi
  done
  module -t list > "$run_dir/modules.txt" 2>&1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Python executes serial sections once; the runner launches an srun MPI step.
"$kernel_python" "$runner" --config "$settings" --out "$run_dir"
