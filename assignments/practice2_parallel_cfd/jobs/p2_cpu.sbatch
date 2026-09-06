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

make serial
make openmp

echo "=== HPC PRACTICE 2 — CPU/OPENMP ==="
echo "USER=$USER"
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "START_TIME=$(date -Is)"

SERIAL_OUT=$(./bin/stencil_serial --n 2048 --steps 200)
echo "=== SERIAL ==="
echo "$SERIAL_OUT"
SERIAL_SECONDS=$(echo "$SERIAL_OUT" | awk -F= '/^PROGRAM_SECONDS=/{print $2}')

OMP1_OUT=$(OMP_NUM_THREADS=1 ./bin/stencil_openmp --n 2048 --steps 200)
echo "=== OPENMP 1 ==="
echo "$OMP1_OUT"
OMP1_SECONDS=$(echo "$OMP1_OUT" | awk -F= '/^PROGRAM_SECONDS=/{print $2}')

OMP4_OUT=$(OMP_NUM_THREADS=4 ./bin/stencil_openmp --n 2048 --steps 200)
echo "=== OPENMP 4 ==="
echo "$OMP4_OUT"
OMP4_SECONDS=$(echo "$OMP4_OUT" | awk -F= '/^PROGRAM_SECONDS=/{print $2}')

OMP16_OUT=$(OMP_NUM_THREADS=16 ./bin/stencil_openmp --n 2048 --steps 200)
echo "=== OPENMP 16 ==="
echo "$OMP16_OUT"
OMP16_SECONDS=$(echo "$OMP16_OUT" | awk -F= '/^PROGRAM_SECONDS=/{print $2}')

CHECKSUM=$(echo "$OMP16_OUT" | awk -F= '/^CHECKSUM=/{print $2}')

echo "=== SUMMARY ==="
echo "SERIAL_SECONDS=$SERIAL_SECONDS"
echo "OMP1_SECONDS=$OMP1_SECONDS"
echo "OMP4_SECONDS=$OMP4_SECONDS"
echo "OMP16_SECONDS=$OMP16_SECONDS"
echo "CHECKSUM=$CHECKSUM"

SIGNATURE=$(printf "%s|%s|%s|cpu\n" "$USER" "$SLURM_JOB_ID" "$(hostname)" | sha256sum | cut -c1-12)
echo "RUN_SIGNATURE=$SIGNATURE"
echo "FINISH_TIME=$(date -Is)"
