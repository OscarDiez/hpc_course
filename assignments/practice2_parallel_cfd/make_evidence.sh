#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 JOB_CPU JOB_MPI JOB_HYBRID [JOB_GPU|NO_SLURM_GPU]" >&2
    exit 2
fi

JOB_CPU=$1
JOB_MPI=$2
JOB_HYBRID=$3
JOB_GPU=${4:-NO_SLURM_GPU}

{
    echo "=== HPC PRACTICE 2 EVIDENCE ==="
    echo "USER=$(whoami)"
    echo "ACCESS_HOST=$(hostname)"
    echo "GENERATED=$(date -Is)"
    echo "JOB_CPU=$JOB_CPU"
    echo "JOB_MPI=$JOB_MPI"
    echo "JOB_HYBRID=$JOB_HYBRID"
    echo "JOB_GPU=$JOB_GPU"

    for spec in "CPU:p2_cpu_${JOB_CPU}.out" "MPI:p2_mpi_${JOB_MPI}.out" "HYBRID:p2_hybrid_${JOB_HYBRID}.out"; do
        label=${spec%%:*}
        file=${spec#*:}
        echo
        echo "=== ${label} OUTPUT ==="
        if [ -f "$file" ]; then cat "$file"; else echo "MISSING_OUTPUT=$file"; fi
    done

    echo
    echo "=== GPU OUTPUT ==="
    if [[ "$JOB_GPU" =~ ^[0-9]+$ ]]; then
        gpu_file="p2_gpu_${JOB_GPU}.out"
        if [ -f "$gpu_file" ]; then cat "$gpu_file"; else echo "MISSING_OUTPUT=$gpu_file"; fi
    else
        echo "GPU_SLURM_OUTPUT=NOT_APPLICABLE"
        echo "GPU_EXTERNAL_ROUTE=$JOB_GPU"
    fi

    for spec in "CPU:$JOB_CPU" "MPI:$JOB_MPI" "HYBRID:$JOB_HYBRID"; do
        label=${spec%%:*}
        jid=${spec#*:}
        echo
        echo "=== SLURM ACCOUNTING — ${label} ==="
        sacct -j "$jid" --format=JobID,JobName,Partition,State,Elapsed,ReqCPUS,AllocCPUS,NodeList || true
    done

    echo
    echo "=== SLURM ACCOUNTING — GPU ==="
    if [[ "$JOB_GPU" =~ ^[0-9]+$ ]]; then
        sacct -j "$JOB_GPU" --format=JobID,JobName,Partition,State,Elapsed,ReqCPUS,AllocCPUS,NodeList || true
    else
        echo "NOT_APPLICABLE=$JOB_GPU"
    fi

    VERIFY=$(printf "%s|%s|%s|%s|%s\n" "$(whoami)" "$JOB_CPU" "$JOB_MPI" "$JOB_HYBRID" "$JOB_GPU" | sha256sum | cut -c1-16)
    echo
    echo "P2_VERIFICATION=$VERIFY"
} | tee p2_evidence.txt
