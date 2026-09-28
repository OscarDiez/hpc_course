#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: bash make_evidence.sh <baseline_job_id> <optimised_job_id>"
    exit 1
fi

BASE_JOB="$1"
OPT_JOB="$2"

BASE_OUT="p3_baseline_$BASE_JOB.out"
OPT_OUT="p3_optimised_$OPT_JOB.out"

for f in "$BASE_OUT" "$OPT_OUT"; do
    if [ ! -f "$f" ]; then
        echo "Missing $f"
        exit 2
    fi
done

{
    echo "PRACTICE=P3_PROFILE_OPTIMISE"
    echo "USER=$USER"
    echo "GENERATED=$(date -Is)"
    echo "BASELINE_JOB_ID=$BASE_JOB"
    echo "OPTIMISED_JOB_ID=$OPT_JOB"
    echo
    echo "=== BASELINE CONFIG ==="
    grep -E '^(HOST|CPU_MODEL|GCC|N|STEPS|REPEATS)=' "$BASE_OUT" || true
    echo
    echo "=== BASELINE RESULT LINES ==="
    grep -E '^(MODE|CHECKSUM|COMPUTE_SECONDS|CHECKPOINT_SECONDS|TOTAL_SECONDS)=' "$BASE_OUT" || true
    echo
    echo "=== OPTIMISED CONFIG ==="
    grep -E '^(HOST|CPU_MODEL|GCC|N|STEPS|REPEATS)=' "$OPT_OUT" || true
    echo
    echo "=== OPTIMISED RESULT LINES ==="
    grep -E '^(MODE|CHECKSUM|COMPUTE_SECONDS|CHECKPOINT_SECONDS|TOTAL_SECONDS)=' "$OPT_OUT" || true
    echo
    echo "=== SOURCE HASH ==="
    sha256sum src/p3_pipeline_student.c
    echo
    echo "=== OUTPUT HASHES ==="
    sha256sum "$BASE_OUT" "$OPT_OUT"
} > p3_evidence.txt

echo "Created p3_evidence.txt"
cat p3_evidence.txt
