#!/bin/bash
# Instructor preflight. No student solution is provided.
set -euo pipefail
BASE='https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m4s5-distributed-ml-2026/assignments/practice4_distributed_ml'
TEST_PARENT="$HOME/m4s5-assignment-validation"
mkdir -p "$TEST_PARENT"
cd "$TEST_PARENT"
curl -fL "$BASE/setup_assignment.sh" -o setup_assignment.sh
TASK_DIR=$(mktemp -d "$TEST_PARENT/check.XXXXXX")
# setup requires a new subdirectory and uses a representative teaching seed.
bash setup_assignment.sh 1042 "$TASK_DIR/work"
cd "$TASK_DIR/work"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
JOB=$(sbatch --parsable run_preflight.sbatch)
JOB=${JOB%%;*}
case "$JOB" in ''|*[!0-9]*) echo "Unexpected sbatch response: $JOB" >&2; exit 2;; esac
echo "Submitted job $JOB. Submission is not PASS."
echo "After completion, run:"
echo "sacct -j $JOB --format=JobID,State,ExitCode,Elapsed"
echo "cat '$PWD/ASSIGNMENT_SUMMARY.txt'"
echo "Log: $PWD/slurm-$JOB.out"
echo "Evidence: $PWD/assignment_report.json"
