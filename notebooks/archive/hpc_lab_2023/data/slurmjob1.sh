#!/bin/bash
#SBATCH --job-name=mpi_job_test      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=email@ufl.edu    # Where to send mail.  Set this to your email address
#SBATCH --ntasks=24                  # Number of MPI tasks (i.e. processes)
#SBATCH --cpus-per-task=1            # Number of cores per MPI task 
#SBATCH --nodes=2                    # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=12         # Maximum number of tasks on each node
#SBATCH --ntasks-per-socket=6        # Maximum number of tasks on each socket
#SBATCH --distribution=cyclic:cyclic # Distribute tasks cyclically first among nodes and then among sockets within a node
#SBATCH --mem-per-cpu=600mb          # Memory (i.e. RAM) per processor
#SBATCH --time=00:05:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=mpi_test_%j.log     # Path to the standard output and error files relative to the working directory

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load intel/2018.1.163 openmpi/3.0.0
srun --mpi=pmix_v1 /data/training/SLURM/prime/prime_mpi