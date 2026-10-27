#!/bin/bash -l
#SBATCH --job-name=m5s1-profile
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:06:00
#SBATCH --output=profile-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR";export LAB_DEVICE=cpu LAB_NEEDS_MPI=0
source scripts/site.sh
out="results/profile-$SLURM_JOB_ID";mkdir -p "$out"
command -v gprof >/dev/null || { echo 'SKIPPED_PROFILE: gprof unavailable'; exit 0; }
# Use the naive version so its explicit copying is visible in a sampling profile.
gcc -O0 -g -pg src/heat_naive.c -o "$out/heat_profile"
cd "$out"
if [[ -x /usr/bin/time ]]; then
 srun -n1 -c1 --cpu-bind=cores /usr/bin/time -v ./heat_profile 512 1000 field.bin >run.log 2>&1
else
 TIMEFORMAT='WALL elapsed_s=%R user_s=%U sys_s=%S maxrss_kb=UNAVAILABLE'
 { time srun -n1 -c1 --cpu-bind=cores ./heat_profile 512 1000 field.bin; } >run.log 2>&1
fi
gprof ./heat_profile gmon.out >gprof.txt
# Keep the binary/code/flags together. Profiling overhead disqualifies these timings from speedup claims.
echo "PROFILE_DONE $PWD"
