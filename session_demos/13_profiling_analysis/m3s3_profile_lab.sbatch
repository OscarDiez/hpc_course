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
for t in gcc mpicc gprof perf strace valgrind scorep scorep-score scalasca /usr/bin/time; do
  if command -v "$t" >/dev/null 2>&1; then
    echo "TOOL name=$t path=$(command -v "$t")"
  else
    echo "TOOL name=$t path=NOT_FOUND"
  fi
done
echo "kernel.perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA|L1d|L2|L3' || true
echo "===== END ENVIRONMENT ====="

gcc -O3 -g -fno-omit-frame-pointer -std=c11 m3s3_profile_demo.c -o m3s3_profile_demo
gcc -O3 -g -pg -fno-omit-frame-pointer -std=c11 m3s3_profile_demo.c -o m3s3_profile_demo_pg
mpicc -O3 -g -std=c11 m3s3_mpi_wait.c -o m3s3_mpi_wait
gcc -O2 -g -std=c11 m3s3_syscall_demo.c -o m3s3_syscall_demo

N=4096
REPS=4

echo "===== BASELINE BAD 5 RUNS ====="
for i in 1 2 3 4 5; do
  echo "BASELINE_RUN run=$i"
  srun -n1 ./m3s3_profile_demo bad "$N" "$REPS"
done
echo "===== END BASELINE BAD 5 RUNS ====="

echo "===== BASELINE GOOD 5 RUNS ====="
for i in 1 2 3 4 5; do
  echo "GOOD_RUN run=$i"
  srun -n1 ./m3s3_profile_demo good "$N" "$REPS"
done
echo "===== END BASELINE GOOD 5 RUNS ====="

echo "===== GNU TIME RESOURCE ANALYSIS ====="
if [ -x /usr/bin/time ]; then
  /usr/bin/time -v srun -n1 ./m3s3_profile_demo bad "$N" "$REPS" >time_program_stdout.txt 2>time_verbose.txt || true
  cat time_verbose.txt
else
  echo "GNU_TIME_STATUS unavailable"
fi
echo "===== END GNU TIME RESOURCE ANALYSIS ====="

echo "===== GPROF INSTRUMENTATION ====="
rm -f gmon.out gprof_report.txt
srun -n1 ./m3s3_profile_demo_pg bad "$N" "$REPS" >/dev/null
if [ -f gmon.out ]; then
  gprof ./m3s3_profile_demo_pg gmon.out >gprof_report.txt || true
  head -n 80 gprof_report.txt
else
  echo "GPROF_STATUS no_gmon_output"
fi
echo "===== END GPROF INSTRUMENTATION ====="

echo "===== STRACE SYSTEM CALL TRACE ====="
if command -v strace >/dev/null 2>&1; then
  strace -c ./m3s3_syscall_demo >strace_program_stdout.txt 2>strace_summary.txt || true
  cat strace_program_stdout.txt
  cat strace_summary.txt
else
  echo "STRACE_STATUS unavailable"
fi
echo "===== END STRACE SYSTEM CALL TRACE ====="

echo "===== CACHEGRIND BAD ====="
if command -v valgrind >/dev/null 2>&1; then
  valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
    ./m3s3_profile_demo bad 1024 1 >cg_bad_stdout.txt 2>cg_bad.txt || true
  grep -E 'I[1L]  misses|D1  misses|LLd misses|LL refs|LL misses|Branches|Mispredicts' cg_bad.txt || true
else
  echo "CACHEGRIND_STATUS unavailable"
fi
echo "===== END CACHEGRIND BAD ====="

echo "===== CACHEGRIND GOOD ====="
if command -v valgrind >/dev/null 2>&1; then
  valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
    ./m3s3_profile_demo good 1024 1 >cg_good_stdout.txt 2>cg_good.txt || true
  grep -E 'I[1L]  misses|D1  misses|LLd misses|LL refs|LL misses|Branches|Mispredicts' cg_good.txt || true
else
  echo "CACHEGRIND_STATUS unavailable"
fi
echo "===== END CACHEGRIND GOOD ====="

echo "===== PERF STAT BAD ====="
if command -v perf >/dev/null 2>&1; then
  set +e
  srun -n1 perf stat -x, -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    ./m3s3_profile_demo bad "$N" "$REPS" >perf_bad_stdout.txt 2>perf_bad_stat.txt
  rc=$?
  set -e
  echo "PERF_STAT_BAD_RC=$rc"
  cat perf_bad_stat.txt
else
  echo "PERF_STAT_BAD_RC=127"
  echo "perf not installed on this node"
fi
echo "===== END PERF STAT BAD ====="

echo "===== PERF SAMPLING ====="
if command -v perf >/dev/null 2>&1; then
  rm -f perf.data perf_report.txt
  set +e
  srun -n1 perf record -F 99 -g -o perf.data -- ./m3s3_profile_demo bad "$N" "$REPS" >/dev/null 2>perf_record_err.txt
  rc=$?
  set -e
  echo "PERF_RECORD_RC=$rc"
  if [ "$rc" -eq 0 ] && [ -f perf.data ]; then
    perf report -i perf.data --stdio --no-children --sort=symbol >perf_report.txt || true
    head -n 80 perf_report.txt
  else
    cat perf_record_err.txt
  fi
else
  echo "PERF_RECORD_RC=127"
  echo "perf not installed on this node"
fi
echo "===== END PERF SAMPLING ====="

echo "===== MPI WAIT TRACE ====="
srun -n4 ./m3s3_mpi_wait
echo "===== END MPI WAIT TRACE ====="

echo "===== SCOREP MPI TRACE ====="
if command -v scorep >/dev/null 2>&1 && command -v scorep-score >/dev/null 2>&1; then
  rm -rf "scorep_m3s3_${SLURM_JOB_ID:-run}"
  scorep mpicc -O2 -g -std=c11 m3s3_mpi_wait.c -o m3s3_mpi_wait_scorep
  export SCOREP_EXPERIMENT_DIRECTORY="scorep_m3s3_${SLURM_JOB_ID:-run}"
  srun -n4 ./m3s3_mpi_wait_scorep >/dev/null
  if [ -f "$SCOREP_EXPERIMENT_DIRECTORY/profile.cubex" ]; then
    scorep-score "$SCOREP_EXPERIMENT_DIRECTORY/profile.cubex" >scorep_score.txt || true
    head -n 80 scorep_score.txt
  else
    echo "SCOREP_STATUS experiment_created_but_profile_missing"
  fi
else
  echo "SCOREP_STATUS unavailable"
fi
echo "===== END SCOREP MPI TRACE ====="

echo "M3S3_PROFILE_LAB_RESULT=PASS"
