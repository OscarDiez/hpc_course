#!/bin/bash -l
#SBATCH --job-name=m5s5-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --time=00:06:00
#SBATCH --output=m5s5-cpu-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./site_env.sh
printf 'JOB_ID=%s HOST=%s DATE=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
source ./environment_probe.sh
lscpu | head -25 || true
python3 run_experiments.py cpu --n "${M5S5_N:-65536}" --queries "${M5S5_QUERIES:-16}"
