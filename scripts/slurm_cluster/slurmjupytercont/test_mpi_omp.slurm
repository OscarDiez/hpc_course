#!/bin/bash
#SBATCH --job-name=mpi_omp_test
#SBATCH --nodes=3               # Request 3 nodes
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH --cpus-per-task=4       # 4 CPUs per task
#SBATCH --time=00:05:00         # Maximum runtime of 5 minutes
#SBATCH --output=mpi_omp_test_output.log
#SBATCH --error=mpi_omp_test_error.log

# Load necessary modules if using module system
# module load mpi/openmpi-x.y.z

# Run MPI program
echo "Running MPI program across nodes..."
mpirun -np 3 ./mpi_hello

# Run OpenMP program on a single node
echo "Running OpenMP program on a single node..."
./omp_hello
