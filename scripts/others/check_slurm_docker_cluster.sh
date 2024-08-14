#!/bin/bash

# Function to submit MPI job
submit_mpi_job() {
    cat <<EOF > mpi_test.slurm
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=test-mpi
#SBATCH --output=mpi_job.out

hostname
EOF
    docker exec slurmctld sbatch mpi_test.slurm
}

# Function to submit OpenMP job
submit_openmp_job() {
    cat <<EOF > openmp_test.slurm
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test-openmp
#SBATCH --output=openmp_job.out

export OMP_NUM_THREADS=4
gcc -fopenmp -o openmp_test openmp_code.c
./openmp_test
EOF
    cat <<EOF > openmp_code.c
#include <omp.h>
#include <stdio.h>
int main() {
    #pragma omp parallel
    {
        printf("Thread %d, Hello World!\\n", omp_get_thread_num());
    }
    return 0;
}
EOF
    docker exec slurmctld sbatch openmp_test.slurm
}

# Function to submit OpenACC job
submit_openacc_job() {
    cat <<EOF > openacc_test.slurm
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=test-openacc
#SBATCH --output=openacc_job.out

pgcc -acc -o openacc_test openacc_code.c
./openacc_test
EOF
    cat <<EOF > openacc_code.c
#include <stdio.h>
#include <openacc.h>
int main() {
    int n = 1000000;
    float a[n], b[n], c[n];
    #pragma acc kernels
    for (int i = 0; i < n; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    #pragma acc kernels
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
    printf("Test completed with c[0] = %f\\n", c[0]);
    return 0;
}
EOF
    docker exec slurmctld sbatch openacc_test.slurm
}

echo "Submitting MPI job..."
submit_mpi_job

echo "Submitting OpenMP job..."
submit_openmp_job

echo "Submitting OpenACC job..."
submit_openacc_job

echo "All jobs submitted, check output files for results."
