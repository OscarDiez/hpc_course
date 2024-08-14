#!/bin/bash

# Description:
# This script installs essential development tools, compilers, MPI libraries, and scientific libraries.
# It is designed to set up a basic HPC environment on a Linux system.
# Usage: Run this script as root or with sudo privileges.

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

# Install additional scientific libraries
echo "Installing additional scientific libraries..."
sudo apt-get install -y libeigen3-dev libboost-all-dev

# Install Environment Modules
echo "Installing Environment Modules..."
sudo apt-get install -y environment-modules
source /etc/profile.d/modules.sh
