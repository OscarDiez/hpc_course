#!/bin/bash
#SBATCH --job-name=p2-mpi
#SBATCH --output=p2_mpi_%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# SciTech CPU-HPC stack: use one coherent compiler/MPI environment.
module purge
module load foss/2023b

make mpi

HOST=$(hostname)

echo "=== HPC PRACTICE 2 — MPI ==="
echo "USER=$USER"
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOST=$HOST"
echo "NODES_USED=$HOST"
echo "MPI_RANKS=$SLURM_NTASKS"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "START_TIME=$(date -Is)"
echo "MPICC=$(command -v mpicc)"
echo "MPIRUN=$(command -v mpirun)"

# SciTech compute nodes currently do not expose srun. Exclude OpenMPI's
# Slurm process launcher and start the four MPI ranks locally inside the
# allocated compute node. The MPI programming model is still distributed
# memory: every rank has its own address space and exchanges halo messages.
mpirun --mca plm ^slurm \
  -np "$SLURM_NTASKS" \
  --bind-to core \
  --mca pml ob1 --mca btl self,vader,tcp \
  ./bin/stencil_mpi --n 2048 --steps 200

SIGNATURE=$(printf "%s|%s|%s|mpi\n" "$USER" "$SLURM_JOB_ID" "$HOST" | sha256sum | cut -c1-12)
echo "RUN_SIGNATURE=$SIGNATURE"
echo "FINISH_TIME=$(date -Is)"
