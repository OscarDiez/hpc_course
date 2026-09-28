#!/bin/bash
#SBATCH --job-name=p3_base
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=p3_baseline_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

N=12000000
STEPS=80
REPEATS=3

echo "=== P3 BASELINE CONFIGURATION ==="
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

make clean >/dev/null 2>&1 || true
make all

echo
echo "=== CORRECTNESS DEMO ==="
./bin/p3_pipeline --demo --mode baseline --checkpoint p3_demo.bin

echo
echo "=== BASELINE REPEATS ==="
for rep in $(seq 1 "$REPEATS"); do
    echo "RUN_BEGIN rep=$rep"
    ./bin/p3_pipeline --mode baseline --n "$N" --steps "$STEPS" \
        --checkpoint "p3_base_$rep.bin"
    echo "RUN_END rep=$rep"
done

echo
echo "=== GPROF PROFILE ==="
./bin/p3_pipeline_gprof --mode baseline --n "$N" --steps "$STEPS" \
    --checkpoint p3_gprof.bin >/dev/null
gprof ./bin/p3_pipeline_gprof gmon.out | head -n 45 || true

echo
echo "=== GNU TIME ==="
if [ -x /usr/bin/time ]; then
    /usr/bin/time -v ./bin/p3_pipeline --mode baseline --n "$N" --steps "$STEPS" \
        --checkpoint p3_time.bin 2>&1 | sed 's/^/TIME /'
else
    echo "TIME unavailable"
fi

echo
echo "=== PERF OPTIONAL ==="
if command -v perf >/dev/null 2>&1; then
    perf stat -e cycles,instructions,cache-references,cache-misses \
        ./bin/p3_pipeline --mode baseline --n "$N" --steps "$STEPS" \
        --checkpoint p3_perf.bin 2>&1 || true
else
    echo "PERF unavailable"
fi

echo
echo "P3_BASELINE_DONE"
