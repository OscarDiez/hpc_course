#!/bin/bash
#SBATCH --job-name=m3s3_profile
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=m3s3_profile-%j.out

set -u
set -o pipefail

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

echo "===== BUILD ====="

gcc -O3 -g -fno-omit-frame-pointer -std=c11 m3s3_profile_demo.c -o m3s3_profile_demo
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "CORE_BUILD_STATUS=FAILED rc=$rc"
  echo "M3S3_PROFILE_LAB_RESULT=FAIL"
  exit 1
fi
echo "CORE_BUILD_STATUS=OK"

gcc -O3 -g -pg -fno-omit-frame-pointer -std=c11 m3s3_profile_demo.c -o m3s3_profile_demo_pg
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "GPROF_BUILD_STATUS=FAILED rc=$rc"
  echo "M3S3_PROFILE_LAB_RESULT=FAIL"
  exit 1
fi
echo "GPROF_BUILD_STATUS=OK"

gcc -O2 -g -std=c11 m3s3_syscall_demo.c -o m3s3_syscall_demo
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "TRACE_BUILD_STATUS=FAILED rc=$rc"
  echo "M3S3_PROFILE_LAB_RESULT=FAIL"
  exit 1
fi
echo "TRACE_BUILD_STATUS=OK"

# MPI is deliberately optional. On SciTech the active EESSI gcc and the
# system OpenMPI wrapper can belong to different software stacks.
MPI_READY=0
if [ -x /usr/bin/mpicc ] && [ -x /usr/bin/gcc ]; then
  echo "Trying MPI build with system OpenMPI wrapper + system gcc"
  OMPI_CC=/usr/bin/gcc /usr/bin/mpicc -O3 -g -std=c11 m3s3_mpi_wait.c -o m3s3_mpi_wait 2>mpi_build.err
  rc=$?
  if [ "$rc" -eq 0 ]; then
    MPI_READY=1
    echo "MPI_BUILD_STATUS=OK"
  else
    echo "MPI_BUILD_STATUS=SKIPPED rc=$rc"
    echo "Reason: MPI toolchain mismatch or unavailable dependencies."
    tail -n 12 mpi_build.err || true
  fi
else
  echo "MPI_BUILD_STATUS=SKIPPED"
fi

echo "===== END BUILD ====="

N=4096
REPS=4

echo "===== BASELINE BAD 5 RUNS ====="
BASELINE_OK=1
for i in 1 2 3 4 5; do
  echo "BASELINE_RUN run=$i"
  srun -n1 ./m3s3_profile_demo bad "$N" "$REPS"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "BASELINE_RUN_FAILED run=$i rc=$rc"
    BASELINE_OK=0
    break
  fi
done
echo "===== END BASELINE BAD 5 RUNS ====="

echo "===== BASELINE GOOD 5 RUNS ====="
GOOD_OK=1
for i in 1 2 3 4 5; do
  echo "GOOD_RUN run=$i"
  srun -n1 ./m3s3_profile_demo good "$N" "$REPS"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "GOOD_RUN_FAILED run=$i rc=$rc"
    GOOD_OK=0
    break
  fi
done
echo "===== END BASELINE GOOD 5 RUNS ====="

if [ "$BASELINE_OK" -ne 1 ] || [ "$GOOD_OK" -ne 1 ]; then
  echo "M3S3_PROFILE_LAB_RESULT=FAIL"
  exit 1
fi

echo "===== GNU TIME RESOURCE ANALYSIS ====="
if [ -x /usr/bin/time ]; then
  /usr/bin/time -v srun -n1 ./m3s3_profile_demo bad "$N" "$REPS" >time_program_stdout.txt 2>time_verbose.txt
  cat time_verbose.txt
else
  echo "GNU_TIME_STATUS=unavailable"
fi
echo "===== END GNU TIME RESOURCE ANALYSIS ====="

echo "===== GPROF INSTRUMENTATION ====="
rm -f gmon.out gprof_report.txt
srun -n1 ./m3s3_profile_demo_pg bad "$N" "$REPS" >/dev/null
rc=$?
if [ "$rc" -eq 0 ] && [ -f gmon.out ]; then
  gprof ./m3s3_profile_demo_pg gmon.out >gprof_report.txt 2>gprof_err.txt
  head -n 80 gprof_report.txt
else
  echo "GPROF_STATUS=unavailable_or_no_gmon rc=$rc"
fi
echo "===== END GPROF INSTRUMENTATION ====="

echo "===== STRACE SYSTEM CALL TRACE ====="
if command -v strace >/dev/null 2>&1; then
  strace -c ./m3s3_syscall_demo >strace_program_stdout.txt 2>strace_summary.txt
  rc=$?
  echo "STRACE_RC=$rc"
  cat strace_program_stdout.txt
  cat strace_summary.txt
else
  echo "STRACE_STATUS=unavailable"
fi
echo "===== END STRACE SYSTEM CALL TRACE ====="

echo "===== CACHEGRIND BAD ====="
if command -v valgrind >/dev/null 2>&1; then
  valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
    ./m3s3_profile_demo bad 1024 1 >cg_bad_stdout.txt 2>cg_bad.txt
  grep -E 'I[1L]  misses|D1  misses|LLd misses|LL refs|LL misses|Branches|Mispredicts' cg_bad.txt || true
else
  echo "CACHEGRIND_STATUS=unavailable"
fi
echo "===== END CACHEGRIND BAD ====="

echo "===== CACHEGRIND GOOD ====="
if command -v valgrind >/dev/null 2>&1; then
  valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
    ./m3s3_profile_demo good 1024 1 >cg_good_stdout.txt 2>cg_good.txt
  grep -E 'I[1L]  misses|D1  misses|LLd misses|LL refs|LL misses|Branches|Mispredicts' cg_good.txt || true
else
  echo "CACHEGRIND_STATUS=unavailable"
fi
echo "===== END CACHEGRIND GOOD ====="

echo "===== PERF STAT BAD ====="
if command -v perf >/dev/null 2>&1; then
  srun -n1 perf stat -x, -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    ./m3s3_profile_demo bad "$N" "$REPS" >perf_bad_stdout.txt 2>perf_bad_stat.txt
  rc=$?
  echo "PERF_STAT_BAD_RC=$rc"
  cat perf_bad_stat.txt
else
  echo "PERF_STAT_BAD_RC=127"
  echo "perf not installed on this node"
fi
echo "===== END PERF STAT BAD ====="

echo "===== PERF STAT GOOD ====="
if command -v perf >/dev/null 2>&1; then
  srun -n1 perf stat -x, -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    ./m3s3_profile_demo good "$N" "$REPS" >perf_good_stdout.txt 2>perf_good_stat.txt
  rc=$?
  echo "PERF_STAT_GOOD_RC=$rc"
  cat perf_good_stat.txt
else
  echo "PERF_STAT_GOOD_RC=127"
  echo "perf not installed on this node"
fi
echo "===== END PERF STAT GOOD ====="

echo "===== PERF SAMPLING ====="
if command -v perf >/dev/null 2>&1; then
  rm -f perf.data perf_report.txt
  srun -n1 perf record -F 99 -g -o perf.data -- ./m3s3_profile_demo bad "$N" "$REPS" >/dev/null 2>perf_record_err.txt
  rc=$?
  echo "PERF_RECORD_RC=$rc"
  if [ "$rc" -eq 0 ] && [ -f perf.data ]; then
    perf report -i perf.data --stdio --no-children --sort=symbol >perf_report.txt 2>/dev/null
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
if [ "$MPI_READY" -eq 1 ]; then
  srun -n4 ./m3s3_mpi_wait
  rc=$?
  echo "MPI_RUN_RC=$rc"
else
  echo "MPI_TRACE_STATUS=unavailable_due_to_MPI_build"
fi
echo "===== END MPI WAIT TRACE ====="

echo "===== SCOREP MPI TRACE ====="
if [ "$MPI_READY" -eq 1 ] && command -v scorep >/dev/null 2>&1 && command -v scorep-score >/dev/null 2>&1; then
  rm -rf "scorep_m3s3_${SLURM_JOB_ID:-run}"
  scorep mpicc -O2 -g -std=c11 m3s3_mpi_wait.c -o m3s3_mpi_wait_scorep
  export SCOREP_EXPERIMENT_DIRECTORY="scorep_m3s3_${SLURM_JOB_ID:-run}"
  srun -n4 ./m3s3_mpi_wait_scorep >/dev/null
  if [ -f "$SCOREP_EXPERIMENT_DIRECTORY/profile.cubex" ]; then
    scorep-score "$SCOREP_EXPERIMENT_DIRECTORY/profile.cubex" >scorep_score.txt
    head -n 80 scorep_score.txt
  else
    echo "SCOREP_STATUS=experiment_created_but_profile_missing"
  fi
else
  echo "SCOREP_STATUS=unavailable"
fi
echo "===== END SCOREP MPI TRACE ====="

echo "M3S3_PROFILE_LAB_RESULT=PASS"
