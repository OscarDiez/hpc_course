#!/bin/bash
#SBATCH --job-name=m3s2_memory
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:08:00
#SBATCH --mem=4G
#SBATCH --output=m3s2_memory-%j.out

set -euo pipefail
if command -v module >/dev/null 2>&1; then module purge; module load foss/2023b; fi
export OMP_NUM_THREADS=${M3_THREADS:-${SLURM_CPUS_PER_TASK:-16}}
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "M3S2_MEMORY_LAB_START job=${SLURM_JOB_ID:-none} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-unknown}"
echo "===== TOPOLOGY ====="
lscpu | grep -E 'Architecture|Model name|Socket|Core|Thread|NUMA|L1d|L2|L3' || true
echo "CACHE_LINE_BYTES=$(getconf LEVEL1_DCACHE_LINESIZE 2>/dev/null || echo unknown)"
if command -v numactl >/dev/null 2>&1; then numactl --hardware || true; fi
echo "===== END TOPOLOGY ====="

gcc -O3 -march=native -fopenmp -std=c11 m3s2_memory_lab.c -o m3s2_memory_lab -lm -lm
rm -f vectorization.log
gcc -O3 -march=native -fopt-info-vec-all=vectorization.log -c m3s2_vectorization.c -o m3s2_vectorization.o || true

echo "===== VECTOR REPORT ====="
if [ -f vectorization.log ]; then
    grep -E 'vectorized|missed|not vectorized|depend' vectorization.log | head -n 40 || cat vectorization.log | head -n 40
else
    echo "No GCC vectorization report produced."
fi
echo "===== END VECTOR REPORT ====="

run_exp(){
  label="$1"; mode="$2"
  echo "===== ${label} ====="
  ./m3s2_memory_lab "$mode"
  echo "===== END ${label} ====="
}

run_exp "EXPERIMENT 1 LATENCY" latency
run_exp "EXPERIMENT 2 BANDWIDTH" stream
run_exp "EXPERIMENT 3 CACHE LINE STRIDE" stride
run_exp "EXPERIMENT 4 ROW VS COLUMN" matrix
run_exp "EXPERIMENT 5 FALSE SHARING" false_sharing
run_exp "EXPERIMENT 6 AOS VS SOA" aossoa
run_exp "EXPERIMENT 7 BLOCKING" tiling
run_exp "EXPERIMENT 8 LOOP FUSION" fusion
run_exp "EXPERIMENT 9 FIRST TOUCH" first_touch
run_exp "EXPERIMENT 10 PREFETCH" prefetch
run_exp "EXPERIMENT 11 SPARSE CSR" sparse
run_exp "EXPERIMENT 12 LOOP UNROLLING" unroll

echo "M3S2_MEMORY_LAB_RESULT=PASS"
