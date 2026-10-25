#!/bin/bash
# Instructor preflight: both required batch sizes; no student solution supplied.
set -euo pipefail
BASE='https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m4s5-distributed-ml-2026/assignments/practice4_distributed_ml'
TEST_PARENT="$HOME/m4s5-assignment-validation"
mkdir -p "$TEST_PARENT"
cd "$TEST_PARENT"
curl -fL "$BASE/setup_assignment.sh" -o setup_assignment.sh
TASK_DIR=$(mktemp -d "$TEST_PARENT/check.XXXXXX")
PREV_JOB=''
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
for BATCH in 128 256; do
    bash "$TEST_PARENT/setup_assignment.sh" 1042 "$TASK_DIR/b$BATCH"
    cd "$TASK_DIR/b$BATCH"
    python3 - "$BATCH" <<'PY'
import json,sys
from pathlib import Path
p=Path('config.json');cfg=json.loads(p.read_text());cfg['global_batch']=int(sys.argv[1]);p.write_text(json.dumps(cfg,indent=2)+'\n')
PY
    SUBMIT=(sbatch --parsable)
    if [ -n "$PREV_JOB" ]; then SUBMIT+=("--dependency=afterany:$PREV_JOB"); fi
    JOB=$("${SUBMIT[@]}" run_baseline.sbatch)
    JOB=${JOB%%;*}
    case "$JOB" in ''|*[!0-9]*) echo "Unexpected sbatch response: $JOB" >&2; exit 2;; esac
    PREV_JOB=$JOB
    echo "Submitted batch-$BATCH preflight job $JOB. Submission is not PASS."
    echo "After completion, run:"
    echo "sacct -j $JOB --format=JobID,State,ExitCode,Elapsed"
    echo "cat '$PWD/ASSIGNMENT_SUMMARY.txt'"
    echo "Log: $PWD/slurm-$JOB.out"
    echo "Evidence: $PWD/assignment_report.json"
done
echo 'Both preflights are submitted; batch 256 waits until batch 128 finishes.'
echo "To print both summaries after completion: cat '$TASK_DIR/b128/ASSIGNMENT_SUMMARY.txt' '$TASK_DIR/b256/ASSIGNMENT_SUMMARY.txt'"
