#!/bin/bash

# Description:
# This script installs GROMACS with MPI support on a Linux system.
# It installs necessary dependencies, downloads GROMACS from source, compiles it with MPI, and configures the environment.
# Usage: Run this script as root or with sudo privileges.

# Verify if GROMACS is already installed
if command -v gmx_mpi &> /dev/null; then
    echo "GROMACS MPI is already installed."
    exit 0
fi

# Update package list and install necessary packages
echo "Installing necessary packages for GROMACS..."
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ gfortran cmake git \
                        libopenmpi-dev openmpi-bin mpich libblas-dev \
                        liblapack-dev libfftw3-dev libhdf5-dev libvtk7-dev \
                        python3 python3-pip

# Set up environment for GROMACS with MPI support
echo "Configuring GROMACS with MPI support..."

# Install GROMACS from source if not already installed
echo "Installing GROMACS from source..."
cd /tmp
wget http://ftp.gromacs.org/pub/gromacs/gromacs-2020.1.tar.gz
tar xfz gromacs-2020.1.tar.gz
cd gromacs-2020.1
mkdir build
cd build
cmake .. -DGMX_MPI=ON -DGMX_BUILD_OWN_FFTW=ON -DGMX_OPENMP=ON -DGMX_DOUBLE=OFF \
         -DGMX_GPU=OFF -DGMX_SIMD=AVX_512 -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs
make -j$(nproc)
sudo make install
cd ~
rm -rf /tmp/gromacs-2020.1

# Create or update modulefile for GROMACS
echo "Updating environment modules..."
MODULEFILES_DIR="/usr/share/modules/modulefiles/gromacs"
sudo mkdir -p $MODULEFILES_DIR
cat <<EOF | sudo tee $MODULEFILES_DIR/2020.1 > /dev/null
#%Module1.0
proc ModulesHelp { } {
    puts stderr "GROMACS 2020.1"
}
module-whatis "GROMACS 2020.1 with MPI support"

set root /usr/local/gromacs

prepend-path PATH \$root/bin
prepend-path LD_LIBRARY_PATH \$root/lib
prepend-path MANPATH \$root/share/man
EOF

# Load the GROMACS module
source /etc/profile.d/modules.sh
module load gromacs/2020.1

# Verify that gmx_mpi is available
if ! command -v gmx_mpi &> /dev/null
then
    echo "GROMACS MPI version is still not found after installation. Exiting."
    exit 1
fi

# Display GROMACS MPI version
gmx_mpi --version

echo "GROMACS with MPI support is installed and configured successfully."
