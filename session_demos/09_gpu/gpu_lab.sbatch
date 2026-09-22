#!/bin/bash
#SBATCH --job-name=m2s4_lab
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=m2s4_lab-%j.out

set -e
cd "$SLURM_SUBMIT_DIR"

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

echo "============================================================"
echo "M2.S4 REAL GPU MINI-LAB"
echo "============================================================"
echo "Host=$(hostname)"
echo "Job=$SLURM_JOB_ID"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

bash -lc '
set -e
module purge
module load nvhpc/25.7
cd "'"$SLURM_SUBMIT_DIR"'"

echo
echo "Compiler:"
echo "nvcc=$(command -v nvcc)"
echo "nvc=$(command -v nvc)"

echo
echo "===== EXPERIMENT 1: CPU VS GPU ====="
nvcc -O3 -Wno-deprecated-gpu-targets quick_compare.cu -o quick_compare
./quick_compare
echo "===== END EXPERIMENT 1 ====="

echo
echo "===== EXPERIMENT 2: CUDA THREAD MAPPING ====="
nvcc -O2 -Wno-deprecated-gpu-targets thread_mapping.cu -o thread_mapping
./thread_mapping
echo "===== END EXPERIMENT 2 ====="

echo
echo "===== EXPERIMENT 3: DATA MOVEMENT ====="
nvcc -O3 -Wno-deprecated-gpu-targets data_movement.cu -o data_movement
./data_movement
echo "===== END EXPERIMENT 3 ====="

echo
echo "===== EXPERIMENT 4: CUDA VECTOR ADD ====="
nvcc -O2 -Wno-deprecated-gpu-targets vector_add.cu -o vector_add
./vector_add
echo "===== END EXPERIMENT 4 ====="

echo
echo "===== EXPERIMENT 5: OPENACC ====="
nvc -O2 -acc -Minfo=accel openacc_vector_add.c -o openacc_vector_add 2>&1 | grep -E "Generating|Accelerator kernel|Tesla|NVIDIA|loop" | head -n 12 || true
./openacc_vector_add
echo "===== END EXPERIMENT 5 ====="
'

echo
echo "M2S4_LAB_RESULT=PASS"
