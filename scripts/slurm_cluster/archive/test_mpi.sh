#!/bin/bash
#SBATCH --job-name=check_hpc_libraries
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=output.log
#SBATCH --error=error.log

# Exit on any error
set -e

# Initialize the module system if not already done
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi

# Load necessary modules for GCC and MPI
module load gcc
module load mpi

# Check for installed libraries
echo "Checking installed HPC libraries and tools:"

# Check if MPI is installed
if mpicc --version &>/dev/null; then
    echo "MPI is installed."
else
    echo "MPI is not installed."
    exit 1
fi

# Check if OpenMP is supported
echo "Checking for OpenMP support:"
if gcc -fopenmp -dM -E - < /dev/null | grep -q _OPENMP; then
    echo "OpenMP is supported by GCC."
else
    echo "OpenMP is not supported by GCC."
    exit 1
fi

# Create a simple MPI/OpenMP program
cat <<EOF >mpi_openmp_test.c
#include <stdio.h>
#include <omp.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int provided, rank, size, tid;

    // Initialize MPI
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print MPI rank and size
    printf("MPI rank: %d out of %d processors\\n", rank, size);

    // OpenMP parallel region
    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        printf("MPI rank: %d, OpenMP thread: %d\\n", rank, tid);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
EOF

# Compile the MPI/OpenMP program
mpicc -fopenmp -o mpi_openmp_test mpi_openmp_test.c

# Run the compiled program using Slurm
echo "Running the MPI/OpenMP program with Slurm..."
srun ./mpi_openmp_test

echo "Script execution completed."

