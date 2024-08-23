#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=result.out
#
#SBATCH --ntasks=6
#
sbcast -f slurm_test.py ./slurm_test.py
srun python3 ./slurm_test.py