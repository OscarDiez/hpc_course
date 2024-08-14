#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=mpi_test_output.log
#SBATCH --error=mpi_test_error.log

# Load the MPI module if necessary
# Uncomment the following line if modules are used
# module load mpi

# Run the MPI job
mpirun -np 3 /home/admin/mpi_hello
