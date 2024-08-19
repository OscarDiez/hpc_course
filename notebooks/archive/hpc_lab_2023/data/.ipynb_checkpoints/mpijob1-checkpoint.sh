#!/bin/bash
#SBATCH --job-name=hybrid_job_test      # Job name
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=email@ufl.edu       # Where to send mail	
#SBATCH --ntasks=8                      # Number of MPI ranks
#SBATCH --cpus-per-task=4               # Number of cores per MPI rank 
#SBATCH --nodes=2                       # Number of nodes
#SBATCH --ntasks-per-node=4             # How many tasks on each node
#SBATCH --ntasks-per-socket=2           # How many tasks on each CPU or socket
#SBATCH --mem-per-cpu=100mb             # Memory per core
#SBATCH --time=00:05:00                 # Time limit hrs:min:sec
#SBATCH --output=hybrid_test_%j.log     # Standard output and error log
pwd; hostname; date
 
module load  intel/2018.1.163  openmpi/3.0.0 raxml/8.2.12
 
srun --mpi=pmix_v1 raxmlHPC-HYBRID-SSE3 -T $SLURM_CPUS_PER_TASK \
      -f a -m GTRGAMMA -s /data/training/SLURM/dna.phy -p $RANDOM \
      -x $RANDOM -N 500 -n dna
 
date