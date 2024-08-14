#!/bin/bash

# Description:
# This script orchestrates the installation of HPC software on multiple Docker containers.
# It copies installation scripts to each container, executes them, and ensures that the environment is correctly set up.
# Usage: Run this script from the Docker host.

# List of containers to execute the script in
containers=(
  "cluster_slurmjupyter_1"
  "cluster_slurmmaster_1"
  "cluster_slurmnode1_1"
  "cluster_slurmnode2_1"
  "cluster_slurmnode3_1"
)

# Iterate over each container and execute the installation scripts
for container in "${containers[@]}"; do
  echo "Copying and executing the installation scripts in $container..."

  # Copy the setup scripts to the shared directory /home/admin inside the container
  docker cp install_hpc_software.sh $container:/home/admin/install_hpc_software.sh
  docker cp install_gromacs.sh $container:/home/admin/install_gromacs.sh

  # Execute the general HPC installation script
  docker exec $container bash -c "sudo bash /home/admin/install_hpc_software.sh"

  # Execute the GROMACS installation script
  docker exec $container bash -c "sudo bash /home/admin/install_gromacs.sh"
done

echo "Setup complete for all containers."
