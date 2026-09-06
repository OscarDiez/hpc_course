#!/bin/bash
#SBATCH --job-name=p2-gpu
#SBATCH --output=p2_gpu_%j.out
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=4G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "=== HPC PRACTICE 2 — GPU ==="
echo "USER=$USER"
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "START_TIME=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if ! command -v nvcc >/dev/null 2>&1; then
    echo "GPU_TOOLCHAIN_STATUS=NVCC_NOT_AVAILABLE"
    echo "The GPU allocation works, but CUDA compiler activation is still required on this cluster."
    exit 3
fi

make cuda

echo "=== CUDA DEMO ==="
./bin/stencil_cuda --demo

echo "=== CUDA LARGE RUN ==="
./bin/stencil_cuda --n 2048 --steps 200

SIGNATURE=$(printf "%s|%s|%s|gpu\n" "$USER" "$SLURM_JOB_ID" "$(hostname)" | sha256sum | cut -c1-12)
echo "RUN_SIGNATURE=$SIGNATURE"
echo "FINISH_TIME=$(date -Is)"
