#!/bin/bash

# Define the setup script content
SETUP_SCRIPT="install_hpc_software.sh"

# Write the installation script to a file
cat << 'EOF' > $SETUP_SCRIPT
#!/bin/bash

# Ensure the script is executed with root privileges
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit
fi

# Update package list and upgrade existing packages
echo "Updating package list and upgrading existing packages..."
apt-get update && apt-get upgrade -y

# Install development tools and compilers
echo "Installing development tools and compilers..."
apt-get install -y build-essential gcc g++ gfortran cmake git

# Install MPI libraries
echo "Installing OpenMPI..."
apt-get install -y libopenmpi-dev openmpi-bin

echo "Installing MPICH..."
apt-get install -y mpich

# Install mathematical libraries
echo "Installing BLAS and LAPACK..."
apt-get install -y libblas-dev liblapack-dev

# Install HDF5
echo "Installing HDF5..."
apt-get install -y libhdf5-dev

# Install VTK (Visualization Toolkit)
echo "Installing VTK..."
apt-get install -y libvtk7-dev

# Install FFTW (Fastest Fourier Transform in the West)
echo "Installing FFTW..."
apt-get install -y libfftw3-dev

# Install GROMACS (Molecular Dynamics)
echo "Installing GROMACS..."
apt-get install -y gromacs

# Install SciPy Stack
echo "Installing Python and SciPy stack..."
apt-get install -y python3 python3-pip
pip3 install --no-cache-dir numpy scipy matplotlib ipython jupyter pandas sympy nose

# Install additional scientific libraries
echo "Installing additional scientific libraries..."
apt-get install -y libeigen3-dev libboost-all-dev

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

echo "Installation complete. Please verify that all tools are correctly installed."
EOF

# List of containers to execute the script in
containers=(
  "cluster_slurmjupyter_1"
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
