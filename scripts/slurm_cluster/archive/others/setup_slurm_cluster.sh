#!/bin/bash

# Stop execution on any error
set -e

# Optional: Update and install Docker Compose if not already installed
# sudo apt-get update
# sudo apt-get install -y docker-compose

# Define the base directory and the Docker Compose directory
BASE_DIR=~/slurm-cluster
COMPOSE_DIR=$BASE_DIR/cluster

# Create the base directory and the Docker Compose directory
mkdir -p $BASE_DIR
mkdir -p $COMPOSE_DIR

# Change to the Docker Compose directory
cd $COMPOSE_DIR

# Check if docker-compose is currently running and clean up if it is
if [ -f "docker-compose.yml" ]; then
  echo "Cleaning up existing Docker Compose deployment..."
  docker-compose down --remove-orphans
  docker-compose rm -fsv
fi

# Create the Docker Compose configuration file with corrected YAML syntax
cat <<EOF >docker-compose.yml
version: '3'
services:
  slurmjupyter:
    image: rancavil/slurm-jupyter:19.05.5-1
    hostname: slurmjupyter
    user: admin
    volumes:
      - shared-vol:/home/admin
    ports:
      - 8888:8888

  slurmmaster:
    image: rancavil/slurm-master:19.05.5-1
    hostname: slurmmaster
    user: admin
    volumes:
      - shared-vol:/home/admin
    ports:
      - 6817:6817
      - 6818:6818
      - 6819:6819

  slurmnode1:
    image: rancavil/slurm-node:19.05.5-1
    hostname: slurmnode1
    user: admin
    volumes:
      - shared-vol:/home/admin
    environment:
      - SLURM_NODENAME=slurmnode1
    links:
      - slurmmaster

  slurmnode2:
    image: rancavil/slurm-node:19.05.5-1
    hostname: slurmnode2
    user: admin
    volumes:
      - shared-vol:/home/admin
    environment:
      - SLURM_NODENAME=slurmnode2
    links:
      - slurmmaster

  slurmnode3:
    image: rancavil/slurm-node:19.05.5-1
    hostname: slurmnode3
    user: admin
    volumes:
      - shared-vol:/home/admin
    environment:
      - SLURM_NODENAME=slurmnode3
    links:
      - slurmmaster

volumes:
  shared-vol:
EOF

# Start the Docker Compose project
docker-compose up -d

# Ensure all containers are up and running
if [ $(docker-compose ps | grep -c 'Up') -ne 5 ]; then
  echo "Error: Not all Docker containers are running."
  exit 1
else
  echo "All Docker containers are running."
fi

# Print successful deployment message
echo "Slurm cluster has been deployed successfully. You can access JupyterLab at http://localhost:8888 (or your VM's IP address if accessing externally)."

