#!/bin/bash

#SBATCH --job-name=m2s1_tasks
#SBATCH --array=0-15
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --output=task_%A_%a.out

python process_task.py ${SLURM_ARRAY_TASK_ID}
