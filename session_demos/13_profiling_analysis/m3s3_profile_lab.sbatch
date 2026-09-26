#!/bin/bash
#SBATCH --job-name=m3s3_profile
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:12:00
#SBATCH --output=m3s3_profile-%j.out

set -u
set -o pipefail

echo "M3S3_PROFILE_LAB_START job=${SLURM_JOB_ID:-none} host=$(hostname)"

echo "===== ENVIRONMENT ====="
for t in gcc gprof strace /usr/bin/time; do
  if command -v "$t" >/dev/null 2>&1; then
    echo "CORE_TOOL name=$t path=$(command -v "$t")"
  else
    echo "CORE_TOOL name=$t path=NOT_FOUND"
  fi
done
echo "CPU model: $(lscpu | awk -F: '/Model name/{gsub(/^[ \t]+/,"",$2); print $2; exit}')"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "EESSI init: /cvmfs/software.eessi.io/versions/2023.06/init/lmod/bash"
echo "===== END ENVIRONMENT ====="

echo "===== BUILD ====="
gcc -O3 -g -fno-omit-frame-pointer -std=c11 m3s3_profile_demo.c -o m3s3_profile_demo
rc=$?
echo "PROFILE_BUILD_RC=$rc"
[ "$rc" -eq 0 ] || { echo "M3S3_PROFILE_LAB_RESULT=FAIL"; exit 1; }

gcc -O3 -g -pg -fno-omit-frame-pointer -std=c11 m3s3_profile_demo.c -o m3s3_profile_demo_pg
rc=$?
echo "GPROF_BUILD_RC=$rc"
[ "$rc" -eq 0 ] || { echo "M3S3_PROFILE_LAB_RESULT=FAIL"; exit 1; }

gcc -O2 -g -std=c11 m3s3_syscall_demo.c -o m3s3_syscall_demo
rc=$?
echo "STRACE_DEMO_BUILD_RC=$rc"
[ "$rc" -eq 0 ] || { echo "M3S3_PROFILE_LAB_RESULT=FAIL"; exit 1; }

gcc -O3 -g -fopenmp -std=c11 m3s3_openmp_wait.c -o m3s3_openmp_wait
rc=$?
echo "OPENMP_BUILD_RC=$rc"
[ "$rc" -eq 0 ] || { echo "M3S3_PROFILE_LAB_RESULT=FAIL"; exit 1; }
echo "===== END BUILD ====="

N=4096
REPS=4

echo "===== BASELINE BAD 5 RUNS ====="
for i in 1 2 3 4 5; do
  echo "BASELINE_RUN run=$i"
  srun -n1 -c1 ./m3s3_profile_demo bad "$N" "$REPS"
done
echo "===== END BASELINE BAD 5 RUNS ====="

echo "===== BASELINE GOOD 5 RUNS ====="
for i in 1 2 3 4 5; do
  echo "GOOD_RUN run=$i"
  srun -n1 -c1 ./m3s3_profile_demo good "$N" "$REPS"
done
echo "===== END BASELINE GOOD 5 RUNS ====="

echo "===== GNU TIME RESOURCE ANALYSIS ====="
/usr/bin/time -v srun -n1 -c1 ./m3s3_profile_demo bad "$N" "$REPS" >time_program_stdout.txt 2>time_verbose.txt
cat time_verbose.txt
echo "===== END GNU TIME RESOURCE ANALYSIS ====="

echo "===== GPROF INSTRUMENTATION ====="
rm -f gmon.out gprof_report.txt pg_run.txt
srun -n1 -c1 ./m3s3_profile_demo_pg bad "$N" "$REPS" >pg_run.txt
rc=$?
echo "GPROF_RUN_RC=$rc"
cat pg_run.txt
if [ "$rc" -eq 0 ] && [ -f gmon.out ]; then
  gprof ./m3s3_profile_demo_pg gmon.out >gprof_report.txt
  head -n 70 gprof_report.txt
else
  echo "GPROF_STATUS=no_gmon_output"
fi
echo "===== END GPROF INSTRUMENTATION ====="

echo "===== STRACE SUMMARY ====="
strace -c ./m3s3_syscall_demo >strace_program_stdout.txt 2>strace_summary.txt
rc=$?
echo "STRACE_SUMMARY_RC=$rc"
cat strace_program_stdout.txt
cat strace_summary.txt
echo "===== END STRACE SUMMARY ====="

echo "===== STRACE EVENT TRACE ====="
strace -tt -T -e trace=openat,write,close,unlink -o strace_events.txt ./m3s3_syscall_demo >/dev/null
rc=$?
echo "STRACE_EVENTS_RC=$rc"
head -n 24 strace_events.txt
echo "===== END STRACE EVENT TRACE ====="

echo "===== OPENMP WAIT TRACE ====="
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  srun -n1 -c4 ./m3s3_openmp_wait
rc=$?
echo "OPENMP_TRACE_RC=$rc"
echo "===== END OPENMP WAIT TRACE ====="

echo "===== EESSI PROFILING TOOLS ====="
EESSI_INIT=/cvmfs/software.eessi.io/versions/2023.06/init/lmod/bash
if [ -r "$EESSI_INIT" ]; then
(
  source "$EESSI_INIT"
  echo "EESSI_STATUS=initialized"
  echo "Available Score-P module:"
  module -t avail Score-P 2>&1 | head -n 20 || true
  echo "Available Extrae module:"
  module -t avail Extrae 2>&1 | head -n 20 || true
  echo "Available CubeLib module:"
  module -t avail CubeLib 2>&1 | head -n 20 || true
)
else
  echo "EESSI_STATUS=init_not_found"
fi
echo "===== END EESSI PROFILING TOOLS ====="

echo "===== SCOREP SERIAL PROFILE ====="
if [ -r "$EESSI_INIT" ]; then
(
  source "$EESSI_INIT"

  SCOREP_READY=0
  if module load Score-P/8.4-gompi-2023b >/dev/null 2>&1; then
    SCOREP_READY=1
    echo "SCOREP_MODULE=Score-P/8.4-gompi-2023b"
    echo "scorep=$(command -v scorep || echo NOT_FOUND)"
    echo "scorep-score=$(command -v scorep-score || echo NOT_FOUND)"
  else
    echo "SCOREP_STATUS=module_load_failed"
  fi

  if [ "$SCOREP_READY" -eq 1 ]; then
    if module load CubeLib/4.8.2-GCCcore-13.2.0 >/dev/null 2>&1; then
      echo "CUBELIB_MODULE=CubeLib/4.8.2-GCCcore-13.2.0"
    else
      echo "CUBELIB_STATUS=module_load_failed"
    fi

    scorep gcc -O2 -g -std=c11 m3s3_profile_demo.c -o m3s3_profile_demo_scorep
    rc=$?
    echo "SCOREP_SERIAL_BUILD_RC=$rc"

    if [ "$rc" -eq 0 ]; then
      rm -rf scorep_serial_bad
      SCOREP_EXPERIMENT_DIRECTORY=scorep_serial_bad \
        srun -n1 -c1 ./m3s3_profile_demo_scorep bad 2048 2 >/dev/null
      rc=$?
      echo "SCOREP_SERIAL_RUN_RC=$rc"

      if [ -f scorep_serial_bad/profile.cubex ]; then
        echo "SCOREP_PROFILE_FILE=scorep_serial_bad/profile.cubex"
        echo "--- scorep-score ---"
        scorep-score scorep_serial_bad/profile.cubex | head -n 60 || true

        if command -v cube_dump >/dev/null 2>&1; then
          echo "--- cube_dump selected functions ---"
          cube_dump -m time -t aggr -z incl -s human scorep_serial_bad/profile.cubex 2>/dev/null \
            | grep -E 'main|compute_bad|initialise|reduction_work|other_work' \
            | head -n 40 || true
        else
          echo "CUBE_DUMP_STATUS=not_found"
        fi
      else
        echo "SCOREP_PROFILE_STATUS=profile_missing"
      fi
    fi
  fi
)
else
  echo "SCOREP_STATUS=eessi_unavailable"
fi
echo "===== END SCOREP SERIAL PROFILE ====="

echo "===== SCOREP OPENMP PROFILE ====="
if [ -r "$EESSI_INIT" ]; then
(
  source "$EESSI_INIT"
  if module load Score-P/8.4-gompi-2023b >/dev/null 2>&1; then
    module load CubeLib/4.8.2-GCCcore-13.2.0 >/dev/null 2>&1 || true

    scorep gcc -O2 -g -fopenmp -std=c11 m3s3_openmp_wait.c -o m3s3_openmp_wait_scorep
    rc=$?
    echo "SCOREP_OPENMP_BUILD_RC=$rc"

    if [ "$rc" -eq 0 ]; then
      rm -rf scorep_openmp_profile
      SCOREP_EXPERIMENT_DIRECTORY=scorep_openmp_profile \
      SCOREP_ENABLE_PROFILING=true \
      SCOREP_ENABLE_TRACING=false \
      OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
        srun -n1 -c4 ./m3s3_openmp_wait_scorep >/dev/null
      rc=$?
      echo "SCOREP_OPENMP_RUN_RC=$rc"

      if [ -f scorep_openmp_profile/profile.cubex ]; then
        echo "SCOREP_OPENMP_PROFILE=scorep_openmp_profile/profile.cubex"
        scorep-score scorep_openmp_profile/profile.cubex | head -n 60 || true

        if command -v cube_dump >/dev/null 2>&1; then
          echo "--- cube_dump OpenMP profile ---"
          cube_dump -m time -t aggr -z incl -s human scorep_openmp_profile/profile.cubex 2>/dev/null \
            | grep -E 'main|do_work|omp|OMP|barrier|parallel' \
            | head -n 60 || true
        fi
      else
        echo "SCOREP_OPENMP_STATUS=profile_missing"
      fi
    fi
  else
    echo "SCOREP_OPENMP_STATUS=module_load_failed"
  fi
)
else
  echo "SCOREP_OPENMP_STATUS=eessi_unavailable"
fi
echo "===== END SCOREP OPENMP PROFILE ====="

echo "===== SCOREP OPENMP TRACE ====="
if [ -r "$EESSI_INIT" ]; then
(
  source "$EESSI_INIT"
  if module load Score-P/8.4-gompi-2023b >/dev/null 2>&1; then
    scorep gcc -O2 -g -fopenmp -std=c11 m3s3_openmp_wait.c -o m3s3_openmp_wait_scorep_trace
    rc=$?
    echo "SCOREP_TRACE_BUILD_RC=$rc"

    if [ "$rc" -eq 0 ]; then
      rm -rf scorep_openmp_trace
      SCOREP_EXPERIMENT_DIRECTORY=scorep_openmp_trace \
      SCOREP_ENABLE_PROFILING=false \
      SCOREP_ENABLE_TRACING=true \
      SCOREP_TOTAL_MEMORY=64M \
      OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
        srun -n1 -c4 ./m3s3_openmp_wait_scorep_trace >/dev/null
      rc=$?
      echo "SCOREP_TRACE_RUN_RC=$rc"

      if [ -d scorep_openmp_trace ]; then
        echo "SCOREP_TRACE_DIRECTORY=scorep_openmp_trace"
        echo "Generated trace files:"
        find scorep_openmp_trace -maxdepth 3 -type f -printf '%p %s bytes\n' | sort | head -n 80
      else
        echo "SCOREP_TRACE_STATUS=trace_directory_missing"
      fi
    fi
  else
    echo "SCOREP_TRACE_STATUS=module_load_failed"
  fi
)
else
  echo "SCOREP_TRACE_STATUS=eessi_unavailable"
fi
echo "===== END SCOREP OPENMP TRACE ====="

echo "M3S3_PROFILE_LAB_RESULT=PASS"
