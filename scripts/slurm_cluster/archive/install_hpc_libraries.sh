#!/bin/bash

# Define the setup script content
SETUP_SCRIPT="install_hpc_software.sh"

# Write the installation script to a file
cat << 'EOF' > $SETUP_SCRIPT
#!/bin/bash

# Update package list and upgrade existing packages
echo "Updating package list and upgrading existing packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install development tools and compilers
echo "Installing development tools and compilers..."
sudo apt-get install -y build-essential gcc g++ gfortran cmake git

# Install MPI libraries
echo "Installing OpenMPI..."
sudo apt-get install -y libopenmpi-dev openmpi-bin

echo "Installing MPICH..."
sudo apt-get install -y mpich

# Install mathematical libraries
echo "Installing BLAS and LAPACK..."
sudo apt-get install -y libblas-dev liblapack-dev

# Install HDF5
echo "Installing HDF5..."
sudo apt-get install -y libhdf5-dev

# Install VTK (Visualization Toolkit)
echo "Installing VTK..."
sudo apt-get install -y libvtk7-dev

# Install FFTW (Fastest Fourier Transform in the West)
echo "Installing FFTW..."
sudo apt-get install -y libfftw3-dev

# Install GROMACS (Molecular Dynamics)
echo "Installing GROMACS..."
sudo apt-get install -y gromacs

# Install SciPy Stack
echo "Installing Python and SciPy stack..."
sudo apt-get install -y python3 python3-pip
pip3 install --no-cache-dir numpy scipy matplotlib ipython jupyter pandas sympy nose

# Install additional scientific libraries
echo "Installing additional scientific libraries..."
sudo apt-get install -y libeigen3-dev libboost-all-dev

# Install Environment Modules
echo "Installing Environment Modules..."
sudo apt-get install -y environment-modules
source /etc/profile.d/modules.sh

# Create modulefiles directory if it doesn't exist
MODULEFILES_DIR="/usr/share/modules/modulefiles"
sudo mkdir -p $MODULEFILES_DIR

# Function to create a modulefile
create_modulefile() {
  local name=$1
  local version=$2
  local path=$3

  MODULEFILE_PATH="$MODULEFILES_DIR/$name"
  sudo mkdir -p $MODULEFILE_PATH

  cat <<EOM | sudo tee $MODULEFILE_PATH/$version > /dev/null
#%Module1.0
proc ModulesHelp { } {
    puts stderr "$name $version"
}
module-whatis "$name $version"

set root $path

prepend-path PATH \$root/bin
prepend-path LD_LIBRARY_PATH \$root/lib
prepend-path MANPATH \$root/share/man
EOM
}

# Create modulefiles for installed tools
create_modulefile "gcc" "10.3.0" "/usr"
create_modulefile "openmpi" "4.1.1" "/usr"
create_modulefile "mpich" "3.3.2" "/usr"
create_modulefile "hdf5" "1.10.6" "/usr"
create_modulefile "vtk" "7.1.1" "/usr"
create_modulefile "fftw" "3.3.8" "/usr"
create_modulefile "gromacs" "2020.1" "/usr"

# Check installation
echo "Checking installed packages..."

# Check for GCC
if gcc --version &>/dev/null; then
    echo "GCC is installed."
else
    echo "GCC installation failed."
fi

# Check for OpenMPI
if mpicc --version &>/dev/null; then
    echo "OpenMPI is installed."
else
    echo "OpenMPI installation failed."
fi

# Check for MPICH
if mpirun --version &>/dev/null; then
    echo "MPICH is installed."
else
    echo "MPICH installation failed."
fi

# Check for HDF5
if h5cc --version &>/dev/null; then
    echo "HDF5 is installed."
else
    echo "HDF5 installation failed."
fi

# Check for VTK
if vtk --version &>/dev/null; then
    echo "VTK is installed."
else
    echo "VTK installation failed."
fi

# Check for GROMACS
if gmx --version &>/dev/null; then
    echo "GROMACS is installed."
else
    echo "GROMACS installation failed."
fi

# Check the number of available CPU cores for MPI
echo "Checking available CPU cores for MPI..."
CPU_CORES=$(nproc)
echo "Available CPU cores: $CPU_CORES"

# Create a test MPI job script that respects the number of available slots
cat << 'EOM' > /home/admin/test_mpi_job.sh
#!/bin/bash
#SBATCH --job-name=test_mpi_job    # Job name
#SBATCH --output=test_mpi_job.out  # Standard output
#SBATCH --error=test_mpi_job.err   # Standard error
#SBATCH --time=00:05:00            # Time limit of 5 minutes
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=$CPU_CORES # Number of CPU cores per task

# Load necessary modules
module load gromacs

# Run a short GROMACS simulation using MPI
mpirun --use-hwthread-cpus -np $CPU_CORES gmx_mpi mdrun -s short_topol.tpr -nsteps 100
EOM

chmod +x /home/admin/test_mpi_job.sh

echo "Installation complete. Please verify that all tools are correctly installed."
EOF

# List of containers to execute the script in
containers=(
  "cluster_slurmjupyter_1"
  "cluster_slurmmaster_1"
  "cluster_slurmnode1_1"
  "cluster_slurmnode2_1"
  "cluster_slurmnode3_1"
)

# Iterate over each container and execute the installation script
for container in "${containers[@]}"; do
  echo "Copying and executing the script in $container..."

  # Copy the script to the shared directory /home/admin inside the container
  docker cp $SETUP_SCRIPT $container:/home/admin/

  # Make the script executable inside the container
  docker exec $container chmod +x /home/admin/$SETUP_SCRIPT

  # Execute the script inside the container with root privileges
  docker exec $container bash -c "sudo /home/admin/$SETUP_SCRIPT"
done

# Clean up
rm $SETUP_SCRIPT

echo "Setup complete for all containers."

