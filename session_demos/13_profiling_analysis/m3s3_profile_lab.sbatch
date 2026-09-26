#!/bin/bash
#SBATCH --job-name=m3s3_profile
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=m3s3_profile-%j.out

set -euo pipefail

echo "M3S3_PROFILE_LAB_START job=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "===== ENVIRONMENT ====="
echo "gcc=$(command -v gcc || true)"
echo "mpicc=$(command -v mpicc || true)"
echo "perf=$(command -v perf || true)"
echo "gprof=$(command -v gprof || true)"
echo "kernel.perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA|L1d|L2|L3' || true
echo "===== END ENVIRONMENT ====="

gcc -O3 -g -fno-omit-frame-pointer -std=c11 m3s3_profile_demo.c -o m3s3_profile_demo
gcc -O3 -g -pg -fno-omit-frame-pointer -std=c11 m3s3_profile_demo.c -o m3s3_profile_demo_pg
mpicc -O3 -g -std=c11 m3s3_mpi_wait.c -o m3s3_mpi_wait

echo "===== BASELINE BAD 5 RUNS ====="
for i in 1 2 3 4 5; do
  echo "BASELINE_RUN run=$i"
  srun -n1 ./m3s3_profile_demo bad
done
echo "===== END BASELINE BAD 5 RUNS ====="

echo "===== BASELINE GOOD 5 RUNS ====="
for i in 1 2 3 4 5; do
  echo "GOOD_RUN run=$i"
  srun -n1 ./m3s3_profile_demo good
done
echo "===== END BASELINE GOOD 5 RUNS ====="

echo "===== GPROF INSTRUMENTATION ====="
rm -f gmon.out
srun -n1 ./m3s3_profile_demo_pg bad >/dev/null
if [ -f gmon.out ]; then
  gprof ./m3s3_profile_demo_pg gmon.out | head -n 80
else
  echo "GPROF_STATUS no_gmon_output"
fi
echo "===== END GPROF INSTRUMENTATION ====="

echo "===== PERF STAT BAD ====="
if command -v perf >/dev/null 2>&1; then
  set +e
  srun -n1 perf stat -x, -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ./m3s3_profile_demo bad >perf_bad_stdout.txt 2>perf_bad_stat.txt
  rc=$?
  set -e
  echo "PERF_STAT_BAD_RC=$rc"
  cat perf_bad_stat.txt
else
  echo "PERF_STAT_BAD_RC=127"
  echo "perf not installed"
fi
echo "===== END PERF STAT BAD ====="

echo "===== PERF STAT GOOD ====="
if command -v perf >/dev/null 2>&1; then
  set +e
  srun -n1 perf stat -x, -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ./m3s3_profile_demo good >perf_good_stdout.txt 2>perf_good_stat.txt
  rc=$?
  set -e
  echo "PERF_STAT_GOOD_RC=$rc"
  cat perf_good_stat.txt
else
  echo "PERF_STAT_GOOD_RC=127"
  echo "perf not installed"
fi
echo "===== END PERF STAT GOOD ====="

echo "===== PERF SAMPLING ====="
if command -v perf >/dev/null 2>&1; then
  rm -f perf.data
  set +e
  srun -n1 perf record -F 99 -g -o perf.data -- ./m3s3_profile_demo bad >/dev/null 2>perf_record_err.txt
  rc=$?
  set -e
  echo "PERF_RECORD_RC=$rc"
  if [ "$rc" -eq 0 ] && [ -f perf.data ]; then
    perf report -i perf.data --stdio --no-children --sort=symbol | head -n 80
  else
    cat perf_record_err.txt
  fi
else
  echo "PERF_RECORD_RC=127"
  echo "perf not installed"
fi
echo "===== END PERF SAMPLING ====="

echo "===== MPI WAIT TRACE ====="
srun -n4 ./m3s3_mpi_wait
echo "===== END MPI WAIT TRACE ====="

echo "M3S3_PROFILE_LAB_RESULT=PASS"
