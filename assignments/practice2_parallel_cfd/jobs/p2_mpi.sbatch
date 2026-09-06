#!/bin/bash
#SBATCH --job-name=p2-mpi
#SBATCH --output=p2_mpi_%j.out
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# SciTech CPU-HPC stack: avoid mixing EESSI compatibility GCC with system OpenMPI.
module purge
module load foss/2023b
module list

make mpi

echo "=== HPC PRACTICE 2 — MPI ==="
echo "USER=$USER"
echo "JOB_ID=$SLURM_JOB_ID"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "NODES_USED=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "START_TIME=$(date -Is)"
echo "MPICC=$(command -v mpicc)"
echo "MPIRUN=$(command -v mpirun)"

mpirun -np "$SLURM_NTASKS" \
  --map-by ppr:2:node \
  --mca pml ob1 --mca btl self,vader,tcp \
  ./bin/stencil_mpi --n 2048 --steps 200

SIGNATURE=$(printf "%s|%s|%s|mpi\n" "$USER" "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" | sha256sum | cut -c1-12)
echo "RUN_SIGNATURE=$SIGNATURE"
echo "FINISH_TIME=$(date -Is)"
