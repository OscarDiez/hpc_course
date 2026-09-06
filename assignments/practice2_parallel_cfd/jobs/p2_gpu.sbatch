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

SHARED_CUDA_IMAGE="${HPC_CUDA_IMAGE:-/data/software/containers/hpc-course-cuda.sif}"

echo "=== HPC PRACTICE 2 — GPU ==="
echo "USER=$USER"
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "START_TIME=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if command -v nvcc >/dev/null 2>&1; then
    echo "GPU_TOOLCHAIN_ROUTE=NATIVE_NVCC"
    echo "NVCC=$(command -v nvcc)"
    make cuda

    echo "=== CUDA DEMO ==="
    ./bin/stencil_cuda --demo

    echo "=== CUDA LARGE RUN ==="
    ./bin/stencil_cuda --n 2048 --steps 200

elif command -v apptainer >/dev/null 2>&1 && [ -r "$SHARED_CUDA_IMAGE" ]; then
    echo "GPU_TOOLCHAIN_ROUTE=APPTAINER_SHARED"
    echo "GPU_CONTAINER=$SHARED_CUDA_IMAGE"
    echo "APPTAINER=$(command -v apptainer)"

    mkdir -p bin

    apptainer exec --nv --pwd "$SLURM_SUBMIT_DIR" "$SHARED_CUDA_IMAGE" bash -lc '
        set -euo pipefail
        echo "NVCC=$(command -v nvcc)"
        nvcc --version | tail -n 4
        nvcc -O3 src/stencil_cuda_student.cu -o bin/stencil_cuda

        echo "=== CUDA DEMO ==="
        ./bin/stencil_cuda --demo

        echo "=== CUDA LARGE RUN ==="
        ./bin/stencil_cuda --n 2048 --steps 200
    '

else
    echo "GPU_TOOLCHAIN_ROUTE=UNAVAILABLE"
    echo "NATIVE_NVCC=NOT_FOUND"
    echo "SHARED_CUDA_IMAGE=$SHARED_CUDA_IMAGE"
    echo "The GPU allocation works, but neither native nvcc nor the shared CUDA container is available."
    exit 3
fi

SIGNATURE=$(printf "%s|%s|%s|gpu\n" "$USER" "$SLURM_JOB_ID" "$(hostname)" | sha256sum | cut -c1-12)
echo "RUN_SIGNATURE=$SIGNATURE"
echo "FINISH_TIME=$(date -Is)"
