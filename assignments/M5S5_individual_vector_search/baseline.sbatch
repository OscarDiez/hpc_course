#!/bin/bash -l
#SBATCH --job-name=m5s5-topk-baseline
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --time=00:06:00
#SBATCH --output=topk-baseline-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./site_env.sh
mkdir -p results
echo "JOB_ID=$SLURM_JOB_ID HOST=$(hostname)"
module -t list 2>&1 || true
gcc --version | head -1
sha256sum topk_baseline.c
gcc -O3 -std=c11 topk_baseline.c -lm -o topk_baseline
for repeat in 1 2 3; do
    srun --exact --exclusive -N1 -n1 -c1 --cpu-bind=cores ./topk_baseline 65536 128 32 10 > "results/baseline-${SLURM_JOB_ID}-${repeat}.json"
done
# This starter deliberately runs ONLY the supplied baseline.
# Add your own candidate build, configurations, output names and correctness checks.
# Do not report these baseline runs as a parallel experiment.
