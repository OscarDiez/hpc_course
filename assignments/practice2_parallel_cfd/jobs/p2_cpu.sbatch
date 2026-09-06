#!/bin/bash
#SBATCH --job-name=p2-cpu
#SBATCH --output=p2_cpu_%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if ! command -v gcc >/dev/null 2>&1; then
    module load foss/2023b
fi

make cpu

echo "=== HPC PRACTICE 2 — CPU/OPENMP ==="
echo "USER=$USER"
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "START_TIME=$(date -Is)"

echo "=== SERIAL ==="
./bin/stencil_serial --n 2048 --steps 200

echo "=== OPENMP 1 ==="
OMP_NUM_THREADS=1 ./bin/stencil_openmp --n 2048 --steps 200

echo "=== OPENMP 4 ==="
OMP_NUM_THREADS=4 ./bin/stencil_openmp --n 2048 --steps 200

echo "=== OPENMP 16 ==="
OMP_NUM_THREADS=16 ./bin/stencil_openmp --n 2048 --steps 200

SIGNATURE=$(printf "%s|%s|%s|cpu\n" "$USER" "$SLURM_JOB_ID" "$(hostname)" | sha256sum | cut -c1-12)
echo "RUN_SIGNATURE=$SIGNATURE"
echo "FINISH_TIME=$(date -Is)"
