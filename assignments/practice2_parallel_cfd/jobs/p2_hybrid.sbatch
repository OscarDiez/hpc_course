#!/bin/bash
#SBATCH --job-name=p2-hybrid
#SBATCH --output=p2_hybrid_%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# SciTech CPU-HPC stack: use one coherent compiler/MPI environment.
module purge
module load foss/2023b

make hybrid
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

HOST=$(hostname)

echo "=== HPC PRACTICE 2 — HYBRID MPI+OPENMP ==="
echo "USER=$USER"
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOST=$HOST"
echo "NODES_USED=$HOST"
echo "MPI_RANKS=$SLURM_NTASKS"
echo "OMP_THREADS_PER_RANK=$OMP_NUM_THREADS"
echo "TOTAL_CPU_THREADS=$((SLURM_NTASKS * OMP_NUM_THREADS))"
echo "START_TIME=$(date -Is)"
echo "MPICC=$(command -v mpicc)"
echo "MPIRUN=$(command -v mpirun)"

# SciTech compute nodes currently do not expose srun. Exclude OpenMPI's
# Slurm process launcher and run the MPI ranks locally inside one allocated
# compute node; OpenMP then provides the intra-process shared-memory level.
mpirun --mca plm ^slurm \
  -np "$SLURM_NTASKS" \
  --bind-to none \
  --mca pml ob1 --mca btl self,vader,tcp \
  ./bin/stencil_hybrid --n 2048 --steps 200

SIGNATURE=$(printf "%s|%s|%s|hybrid\n" "$USER" "$SLURM_JOB_ID" "$HOST" | sha256sum | cut -c1-12)
echo "RUN_SIGNATURE=$SIGNATURE"
echo "FINISH_TIME=$(date -Is)"
