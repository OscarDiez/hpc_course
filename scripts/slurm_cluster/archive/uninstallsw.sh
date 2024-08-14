#!/bin/bash

# Ensure the script is executed with root privileges
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit
fi

echo "Starting uninstallation of HPC libraries and tools..."

# Uninstall development tools and compilers
echo "Uninstalling development tools and compilers..."
apt-get remove --purge -y build-essential gcc g++ gfortran cmake git

# Uninstall MPI libraries
echo "Uninstalling OpenMPI..."
apt-get remove --purge -y libopenmpi-dev openmpi-bin

echo "Uninstalling MPICH..."
apt-get remove --purge -y mpich

# Uninstall mathematical libraries
echo "Uninstalling BLAS and LAPACK..."
apt-get remove --purge -y libblas-dev liblapack-dev

# Uninstall HDF5
echo "Uninstalling HDF5..."
apt-get remove --purge -y libhdf5-dev

# Uninstall VTK (Visualization Toolkit)
echo "Uninstalling VTK..."
apt-get remove --purge -y libvtk7-dev

# Uninstall FFTW (Fastest Fourier Transform in the West)
echo "Uninstalling FFTW..."
apt-get remove --purge -y libfftw3-dev

# Uninstall GROMACS (Molecular Dynamics)
echo "Uninstalling GROMACS..."
apt-get remove --purge -y gromacs

# Uninstall SciPy Stack and Python
echo "Uninstalling Python and SciPy stack..."
apt-get remove --purge -y python3 python3-pip
pip3 uninstall -y numpy scipy matplotlib ipython jupyter pandas sympy nose

# Uninstall additional scientific libraries
echo "Uninstalling additional scientific libraries..."
apt-get remove --purge -y libeigen3-dev libboost-all-dev

# Uninstall Environment Modules
echo "Uninstalling Environment Modules..."
apt-get remove --purge -y environment-modules

# Clean up any remaining configuration files and dependencies
echo "Cleaning up..."
apt-get autoremove -y
apt-get autoclean -y

# Remove modulefiles directory if it exists
MODULEFILES_DIR="/usr/share/modules/modulefiles"
if [ -d "$MODULEFILES_DIR" ]; then
  echo "Removing modulefiles directory..."
  rm -rf $MODULEFILES_DIR
fi

echo "Uninstallation complete. The system should be free of the previously installed HPC libraries and tools."
