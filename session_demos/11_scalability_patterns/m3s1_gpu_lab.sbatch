#!/bin/bash
#SBATCH --job-name=m3s1-gpu-map
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:03:00
#SBATCH --output=m3s1_gpu-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

module purge
module load nvhpc/25.7

echo '===== GPU MAP: KERNEL TIME VS END-TO-END TIME ====='
nvidia-smi --query-gpu=name --format=csv,noheader
nvcc -O3 -Wno-deprecated-gpu-targets m3s1_cuda_map.cu -o m3s1_cuda_map
./m3s1_cuda_map
echo 'M3S1_GPU_LAB_RESULT=PASS'
