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
set -eu
cd "$SLURM_SUBMIT_DIR"
module load nvhpc/25.7
module list 2>gpu-modules.txt
nvidia-smi
# Compile the same source. The _OPENACC blocks become active in this build.
nvc -O2 -acc -gpu=ccnative -Minfo=accel heat.c -o heat_acc
./heat_acc "${N:-128}" "${STEPS:-150}" openacc.bin > gpu-time.csv
echo GPU_DONE
