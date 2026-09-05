#!/bin/bash
#SBATCH --job-name=hpc-openacc
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=gpu-openacc-%j.out

set -euo pipefail

cd "$(dirname "$0")"
bash ./run_openacc_demo.sh
