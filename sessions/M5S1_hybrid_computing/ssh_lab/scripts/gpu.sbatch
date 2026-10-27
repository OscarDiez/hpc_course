#!/bin/bash -l
#SBATCH --job-name=m5s1-heat-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=2G
#SBATCH --time=00:06:00
#SBATCH --output=gpu-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR";export LAB_DEVICE=gpu
# The GPU compiler stack is selected inside the GPU allocation.
source scripts/site.sh
n=${N:-512};steps=${STEPS:-500};repeats=${REPEATS:-3}
out="results/gpu-$SLURM_JOB_ID";mkdir -p "$out"
{
 date -u;hostname;scontrol show job "$SLURM_JOB_ID";nvidia-smi;nvcc --version
 if type module >/dev/null 2>&1; then module list; fi
 sha256sum src/* scripts/*.sh scripts/*.py scripts/*.sbatch
} >"$out/environment.txt" 2>&1
# Compile on the assigned node and keep compiler diagnostics with this run.
bash scripts/build.sh gpu
sha256sum bin/* >"$out/binary-hashes.txt"
bash scripts/measure.sh reference 1 1 bin/heat_serial "$n" "$steps" "$out" 0
# The serial reference uses this same node, avoiding a different CPU model.
for mode in serial cuda openacc; do
 echo "STEP $mode: execute and validate"
 case "$mode" in serial) exe=bin/heat_serial;;cuda) exe=bin/heat_cuda;;openacc)
 if [[ ! -x bin/heat_acc ]]; then echo 'SKIPPED OpenACC: compiler unavailable';continue;fi
 exe=bin/heat_acc;;esac
 for ((r=0;r<=repeats;r++));do
  bash scripts/measure.sh "$mode" 1 1 "$exe" "$n" "$steps" "$out" "$r"
  python3 scripts/validate.py "$out/reference-r0.bin" "$out/$mode-r$r.bin" "$n"
 done
 echo "CHECKPOINT PASS $mode"
done
python3 scripts/summarize.py "$out"
echo "GPU_GUIDED_LAB_DONE $out"
