#!/bin/bash -l
#SBATCH --job-name=m5s1-openacc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:06:00
#SBATCH --output=gpu-%j.out
set -euo pipefail
# Slurm starts this script in the submitted/requested working folder.
N=${N:-128}; STEPS=${STEPS:-150}
module load nvhpc/25.7
module list
nvidia-smi
nvc -O2 -acc -gpu=ccnative -Minfo=accel heat_openacc.c -o openacc
# Make a CPU reference with exactly the same inputs in this job.
# Reuse the CPU executable compiled in the serial step.
out="$PWD/results/gpu-$SLURM_JOB_ID"
mkdir -p "$out"
./serial "$N" "$STEPS" "$out/reference.vtk"
./openacc "$N" "$STEPS" "$out/openacc.vtk"
# Allow tiny rounding differences, comparing all scalar values after the VTK header.
paste "$out/reference.vtk" "$out/openacc.vtk" | awk '
NR>10 { difference=$1-$2; if(difference<0) difference=-difference;
        if(difference>maximum) maximum=difference }
END { print "Maximum GPU error:",maximum+0; exit(maximum>1e-10) }'
printf '%s\n' "$out" > latest_gpu.txt
echo GPU_PASS
