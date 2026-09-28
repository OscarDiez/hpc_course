#!/bin/bash
#SBATCH --job-name=p3_opt
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=p3_optimised_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

N=12000000
STEPS=80
REPEATS=3

echo "=== P3 OPTIMISED CONFIGURATION ==="
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
echo "CPU_MODEL=$(lscpu | awk -F: '/Model name/ {gsub(/^[ \\t]+/, "", $2); print $2; exit}')"
echo "GCC=$(gcc --version | head -n 1)"
echo "N=$N"
echo "STEPS=$STEPS"
echo "REPEATS=$REPEATS"
echo "MODULES_BEGIN"
module -t list 2>&1 || true
echo "MODULES_END"

make all

echo
echo "=== CORRECTNESS DEMO: BASELINE ==="
./bin/p3_pipeline --demo --mode baseline --checkpoint p3_demo_base.bin

echo
echo "=== CORRECTNESS DEMO: OPTIMISED ==="
./bin/p3_pipeline --demo --mode optimised --checkpoint p3_demo_opt.bin

echo
echo "=== OPTIMISED REPEATS ==="
for rep in $(seq 1 "$REPEATS"); do
    echo "RUN_BEGIN rep=$rep"
    ./bin/p3_pipeline --mode optimised --n "$N" --steps "$STEPS" \
        --checkpoint "p3_opt_$rep.bin"
    echo "RUN_END rep=$rep"
done

echo
echo "=== GPROF OPTIMISED ==="
rm -f gmon.out
./bin/p3_pipeline_gprof --mode optimised --n "$N" --steps "$STEPS" \
    --checkpoint p3_gprof_opt.bin >/dev/null
gprof ./bin/p3_pipeline_gprof gmon.out | head -n 45 || true

echo
echo "=== PERF OPTIONAL ==="
if command -v perf >/dev/null 2>&1; then
    perf stat -e cycles,instructions,cache-references,cache-misses \
        ./bin/p3_pipeline --mode optimised --n "$N" --steps "$STEPS" \
        --checkpoint p3_perf_opt.bin 2>&1 || true
else
    echo "PERF unavailable"
fi

echo
echo "P3_OPTIMISED_DONE"
