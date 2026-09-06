#!/bin/bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 JOB_CPU JOB_MPI JOB_HYBRID JOB_GPU" >&2
    exit 2
fi

JOB_CPU=$1
JOB_MPI=$2
JOB_HYBRID=$3
JOB_GPU=$4

{
    echo "=== HPC PRACTICE 2 EVIDENCE ==="
    echo "USER=$(whoami)"
    echo "ACCESS_HOST=$(hostname)"
    echo "GENERATED=$(date -Is)"
    echo "JOB_CPU=$JOB_CPU"
    echo "JOB_MPI=$JOB_MPI"
    echo "JOB_HYBRID=$JOB_HYBRID"
    echo "JOB_GPU=$JOB_GPU"

    for spec in "CPU:p2_cpu_${JOB_CPU}.out" "MPI:p2_mpi_${JOB_MPI}.out" "HYBRID:p2_hybrid_${JOB_HYBRID}.out" "GPU:p2_gpu_${JOB_GPU}.out"; do
        label=${spec%%:*}
        file=${spec#*:}
        echo
        echo "=== ${label} OUTPUT ==="
        if [ -f "$file" ]; then cat "$file"; else echo "MISSING_OUTPUT=$file"; fi
    done

    for spec in "CPU:$JOB_CPU" "MPI:$JOB_MPI" "HYBRID:$JOB_HYBRID" "GPU:$JOB_GPU"; do
        label=${spec%%:*}
        jid=${spec#*:}
        echo
        echo "=== SLURM ACCOUNTING — ${label} ==="
        sacct -j "$jid" --format=JobID,JobName,Partition,State,Elapsed,ReqCPUS,AllocCPUS,NodeList || true
    done

    VERIFY=$(printf "%s|%s|%s|%s|%s\n" "$(whoami)" "$JOB_CPU" "$JOB_MPI" "$JOB_HYBRID" "$JOB_GPU" | sha256sum | cut -c1-16)
    echo
    echo "P2_VERIFICATION=$VERIFY"
} | tee p2_evidence.txt
