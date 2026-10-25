#!/bin/bash -l
#SBATCH --job-name=m5s5-replay
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:03:00
#SBATCH --output=m5s5-replay-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./site_env.sh
printf 'JOB_ID=%s HOST=%s DATE=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
source ./environment_probe.sh
lscpu | head -25 || true
python3 container_workflow.py
