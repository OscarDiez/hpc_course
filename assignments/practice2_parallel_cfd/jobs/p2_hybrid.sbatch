#!/bin/bash
#SBATCH --job-name=p2-hybrid
#SBATCH --output=p2_hybrid_%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if ! command -v mpicc >/dev/null 2>&1; then
    module load foss/2023b
fi

make hybrid
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "=== HPC PRACTICE 2 — HYBRID MPI+OPENMP ==="
echo "USER=$USER"
echo "JOB_ID=$SLURM_JOB_ID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "NODES_USED=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)"
echo "MPI_RANKS=$SLURM_NTASKS"
echo "OMP_THREADS_PER_RANK=$OMP_NUM_THREADS"
echo "TOTAL_CPU_THREADS=$((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "START_TIME=$(date -Is)"

srun ./bin/stencil_hybrid --n 2048 --steps 200

SIGNATURE=$(printf "%s|%s|%s|hybrid\n" "$USER" "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" | sha256sum | cut -c1-12)
echo "RUN_SIGNATURE=$SIGNATURE"
echo "FINISH_TIME=$(date -Is)"
