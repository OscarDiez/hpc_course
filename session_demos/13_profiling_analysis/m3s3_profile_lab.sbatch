#!/bin/bash
#SBATCH --job-name=m3s3_profile
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:08:00
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
head -n 20 strace_events.txt
echo "===== END STRACE EVENT TRACE ====="

echo "===== OPENMP WAIT TRACE ====="
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  srun -n1 -c4 ./m3s3_openmp_wait
rc=$?
echo "OPENMP_TRACE_RC=$rc"
echo "===== END OPENMP WAIT TRACE ====="

echo "M3S3_PROFILE_LAB_RESULT=PASS"
