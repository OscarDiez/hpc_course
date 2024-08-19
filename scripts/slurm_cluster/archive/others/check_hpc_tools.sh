#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print an error message
function error_msg {
    echo "ERROR: $1" 1>&2
}

# Function to check service status
check_service() {
    if docker exec slurmctld pgrep "$1" &> /dev/null; then
        echo "$1 is running."
    else
        echo "$1 is not running."
        error_msg "$1 failed to start"
        exit 1
    fi
}

# Check Slurm services
echo "Checking Slurm services..."
check_service "slurmctld"
check_service "slurmd"

# Check MPI installation by compiling and running a simple program
echo "Checking MPI installation..."
mpi_code="#include <mpi.h>\n#include <stdio.h>\nint main(int argc, char** argv) {\nMPI_Init(&argc, &argv);\nprintf(\"MPI test program executed successfully\\n\");\nMPI_Finalize();\nreturn 0;}"
echo -e "$mpi_code" > mpi_test.c
docker exec slurmctld mpicc mpi_test.c -o mpi_test
docker exec slurmctld mpirun -np 2 ./mpi_test

# Check OpenMP by compiling and running a simple program
echo "Checking OpenMP installation..."
openmp_code="#include <omp.h>\n#include <stdio.h>\nint main() {\n#pragma omp parallel\n{printf(\"OpenMP thread %d\\n\", omp_get_thread_num());}\nreturn 0;}"
echo -e "$openmp_code" > openmp_test.c
docker exec slurmctld gcc -fopenmp openmp_test.c -o openmp_test
docker exec slurmctld ./openmp_test

# Submit a simple job to Slurm using MPI on 1 node
echo "Submitting a simple MPI job on 1 node..."
echo -e "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=2\nmpirun ./mpi_test" > mpi_job_1node.sh
docker cp mpi_job_1node.sh slurmctld:/mpi_job_1node.sh
docker exec slurmctld sbatch /mpi_job_1node.sh

# Submit a simple job to Slurm using MPI on 2 nodes
echo "Submitting a simple MPI job on 2 nodes..."
echo -e "#!/bin/bash\n#SBATCH --nodes=2\n#SBATCH --ntasks=4\nmpirun ./mpi_test" > mpi_job_2nodes.sh
docker cp mpi_job_2nodes.sh slurmctld:/mpi_job_2nodes.sh
docker exec slurmctld sbatch /mpi_job_2nodes.sh

# Check additional tools
echo "Checking additional tools..."
docker exec slurmctld gcc --version
docker exec slurmctld gfortran --version
docker exec slurmctld python --version

echo "Check completed. Jobs have been submitted to Slurm."
