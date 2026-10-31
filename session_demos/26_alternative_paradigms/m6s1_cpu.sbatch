#!/bin/bash
#SBATCH --job-name=m6s1-concepts
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=m6s1-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
# Existing course stack. If modules are unavailable, use the inherited GCC.
if type module >/dev/null 2>&1; then
    module load foss/2023b || { echo 'FAIL: foss/2023b unavailable; inspect module avail'; exit 2; }
fi
command -v python3 >/dev/null || { echo 'FAIL: python3 missing'; exit 2; }
export OMP_NUM_THREADS=1
python3 run_suite.py --settings settings.json --output results.json
