#!/bin/bash
# Freeze all code and scripts into each submission. Queued jobs never read edited files.
set -euo pipefail
cd "$(dirname "$0")/.."
device=${1:-cpu};stage=${2:-all}
case "$device" in cpu|gpu|paraview|profile) ;; *) echo 'Use cpu, gpu, paraview or profile' >&2;exit 2;;esac
stamp=$(date -u +%Y%m%dT%H%M%S)-$$
run="$PWD/runs/$device-$stage-$stamp"
mkdir -p "$run";cp -r src scripts "$run/"
cd "$run"
# A notebook/interactive Slurm session can export incompatible memory variables.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
export STAGE=$stage
export MPI_LAUNCHER=${MPI_LAUNCHER:-auto}
job=$(sbatch --parsable "scripts/$device.sbatch");job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch output: $job" >&2;exit 2; }
printf '%s\n' "$job" >job_id.txt
printf '%s\n' "$PWD" >../../last_run_path.txt
printf 'Job %s\nFrozen run: %s\nCheck: squeue -j %s\n' "$job" "$PWD" "$job"
