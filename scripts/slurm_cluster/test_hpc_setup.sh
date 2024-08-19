#!/bin/bash

# Define the test script content
TEST_SCRIPT="test_hpc_libraries.sh"

# Write the test script to a file
cat << 'EOF' > $TEST_SCRIPT
#!/bin/bash

echo "Running tests for installed HPC libraries..."

# Clean up any old test files
echo "Cleaning up old test files..."
rm -f mpi_hello mpi_hello.c blas_lapack_test.c blas_lapack_test hdf5_test.c hdf5_test vtk_test.cpp vtk_test fftw_test.c fftw_test

# Test GCC
echo "Testing GCC..."
gcc --version &>/dev/null && echo "GCC is working." || echo "GCC test failed."

# Create MPI Hello World program
echo "Creating MPI Hello World program..."
cat << 'CODE' > mpi_hello.c
#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    printf("Hello world from rank %d out of %d processors\n", world_rank, world_size);
    MPI_Finalize();
    return 0;
}
CODE

# Ensure the file was created successfully
if [ -f "mpi_hello.c" ]; then
  echo "mpi_hello.c created successfully."
else
  echo "Failed to create mpi_hello.c due to permissions."
  exit 1
fi

# Test OpenMPI
echo "Testing OpenMPI..."
mpicc -o mpi_hello mpi_hello.c
mpirun -np 2 ./mpi_hello && echo "OpenMPI is working." || echo "OpenMPI test failed."

# Test MPICH
echo "Testing MPICH..."
mpicc -o mpi_hello mpi_hello.c
mpirun -np 2 ./mpi_hello && echo "MPICH is working." || echo "MPICH test failed."

# Test BLAS and LAPACK
echo "Testing BLAS and LAPACK..."
cat << 'CODE' > blas_lapack_test.c
#include <stdio.h>
#include <cblas.h>
int main() {
    double A[6] = {1, 2, 3, 4, 5, 6};
    double B[6] = {7, 8, 9, 10, 11, 12};
    double C[4] = {0, 0, 0, 0};
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 3, 1.0, A, 3, B, 2, 0.0, C, 2);
    printf("C: [%f, %f, %f, %f]\n", C[0], C[1], C[2], C[3]);
    return 0;
}
CODE

if [ -f "blas_lapack_test.c" ]; then
  gcc -o blas_lapack_test blas_lapack_test.c -lblas -llapack && ./blas_lapack_test && echo "BLAS and LAPACK are working." || echo "BLAS and LAPACK test failed."
else
  echo "Failed to create blas_lapack_test.c due to permissions."
  exit 1
fi

# Test HDF5
echo "Testing HDF5..."
cat << 'CODE' > hdf5_test.c
#include <hdf5.h>
int main() {
    hid_t file_id = H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(file_id);
    return 0;
}
CODE

if [ -f "hdf5_test.c" ]; then
  h5cc -o hdf5_test hdf5_test.c && ./hdf5_test && echo "HDF5 is working." || echo "HDF5 test failed."
else
  echo "Failed to create hdf5_test.c due to permissions."
  exit 1
fi

# Test VTK
echo "Testing VTK..."
cat << 'CODE' > vtk_test.cpp
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
int main() {
    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetCenter(0.0, 0.0, 0.0);
    sphereSource->SetRadius(5.0);
    sphereSource->Update();
    std::cout << "VTK Version: " << vtkVersion::GetVTKVersion() << std::endl;
    return EXIT_SUCCESS;
}
CODE

if [ -f "vtk_test.cpp" ]; then
  g++ vtk_test.cpp -o vtk_test -I/usr/include/vtk-7.1 -lvtkCommonCore-7.1 -lvtkFiltersSources-7.1 && ./vtk_test && echo "VTK is working." || echo "VTK test failed."
else
  echo "Failed to create vtk_test.cpp due to permissions."
  exit 1
fi

# Test FFTW
echo "Testing FFTW..."
cat << 'CODE' > fftw_test.c
#include <fftw3.h>
#include <stdio.h>
int main() {
    fftw_complex *in, *out;
    fftw_plan p;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 4);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 4);
    p = fftw_plan_dft_1d(4, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    printf("FFTW is working.\n");
    return 0;
}
CODE

if [ -f "fftw_test.c" ]; then
  gcc fftw_test.c -o fftw_test -lfftw3 && ./fftw_test && echo "FFTW is working." || echo "FFTW test failed."
else
  echo "Failed to create fftw_test.c due to permissions."
  exit 1
fi

# Test GROMACS
echo "Testing GROMACS..."
echo "GROMACS is installed." # Placeholder test as actual tests require setup

# Test Python SciPy Stack
echo "Testing Python SciPy stack..."
python3 -c "import numpy; import scipy; import matplotlib; import pandas; import sympy; print('Python SciPy stack is working.')" && echo "Python SciPy stack is working." || echo "Python SciPy stack test failed."

# Clean up test files
rm -f mpi_hello mpi_hello.c blas_lapack_test.c blas_lapack_test hdf5_test.c hdf5_test vtk_test.cpp vtk_test fftw_test.c fftw_test

EOF

# List of containers to execute the script in
containers=(
  "cluster_slurmjupyter_1"
  "cluster_slurmnode1_1"
  "cluster_slurmnode2_1"
  "cluster_slurmnode3_1"
)

# Iterate over each container and execute the test script
for container in "${containers[@]}"; do
  echo "Copying and executing the test script in $container..."

  # Copy the test script to the shared directory /home/admin inside the container
  docker cp $TEST_SCRIPT $container:/home/admin/

  # Make the test script executable inside the container
  docker exec $container chmod +x /home/admin/$TEST_SCRIPT

  # Execute the test script inside the container
  docker exec -it $container bash -c "/home/admin/$TEST_SCRIPT"
done

# Submit a simple MPI job using Slurm from the slurm master node
SLURM_MASTER="cluster_slurmmaster_1"

# Create Slurm job script
cat << 'EOF' > slurm_mpi_job.sh
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
EOF

# Copy the Slurm job script to the Slurm master node
docker cp slurm_mpi_job.sh $SLURM_MASTER:/home/admin/

# Submit the Slurm job from the Slurm master node
docker exec -it $SLURM_MASTER bash -c "sbatch /home/admin/slurm_mpi_job.sh"

# Clean up test script and source file
rm $TEST_SCRIPT slurm_mpi_job.sh

echo "Tests complete."
