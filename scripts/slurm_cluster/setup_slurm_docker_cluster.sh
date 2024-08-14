#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print an error message and exit
function error_exit {
    echo "$1" 1>&2
    exit 1
}

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    error_exit "Docker is not installed. Please install Docker first."
elif ! command -v docker-compose &> /dev/null; then
    error_exit "Docker Compose is not installed. Please install Docker Compose first."
fi

# Check if the Slurm Docker Cluster repository exists and clean up previous installations
if [ -d "slurm-docker-cluster" ]; then
    echo "Existing Slurm Docker Cluster found. Cleaning up..."
    cd slurm-docker-cluster
    docker-compose down -v
    cd ..
    rm -rf slurm-docker-cluster
fi

# Clone the Slurm Docker Cluster repository
echo "Cloning the Slurm Docker Cluster repository..."
git clone https://github.com/giovtorres/slurm-docker-cluster.git
cd slurm-docker-cluster

# Update slurm.conf to handle three compute nodes
echo "Updating Slurm configuration..."
sed -i '/NodeName=c1/ a NodeName=c2 CPUs=1 State=UNKNOWN\nNodeName=c3 CPUs=1 State=UNKNOWN' etc/slurm/slurm.conf

# Adding JupyterHub service to docker-compose.yml
echo "Configuring JupyterHub service..."
cat <<EOF >> docker-compose.yml
  jupyterhub:
    image: jupyterhub/jupyterhub:latest
    volumes:
      - ./jupyterhub:/srv/jupyterhub
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - JUPYTERHUB_CRYPT_KEY=\$(openssl rand -hex 32)
    ports:
      - "8000:8000"
    depends_on:
      - slurmctld
EOF

# Create JupyterHub config
mkdir -p jupyterhub
cat <<EOF > jupyterhub/jupyterhub_config.py
c = get_config()
c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.DockerSpawner.image = 'jupyter/scipy-notebook:latest'
c.JupyterHub.hub_ip = 'jupyterhub'
c.JupyterHub.hub_connect_ip = 'jupyterhub'
c.JupyterHub.port = 8000
EOF

# Modify Dockerfile to install additional libraries if needed
echo "Updating Dockerfile..."
cat <<EOF > Dockerfile
FROM rockylinux:8
RUN yum -y install epel-release && \\
    yum -y groupinstall "Development Tools" && \\
    yum -y install openmpi-devel gcc-gfortran openssl-devel papi-devel fftw-devel environment-modules wget
RUN yum -y install hdf5-devel || { \\
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz && \\
    tar -xzvf hdf5-1.10.6.tar.gz && \\
    cd hdf5-1.10.6 && \\
    ./configure --prefix=/usr/local && \\
    make && make install; }
LABEL org.opencontainers.image.source="https://github.com/giovtorres/slurm-docker-cluster"
EOF

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Start the cluster
echo "Starting the cluster with three compute nodes and JupyterHub..."
docker-compose up -d

# Wait for slurmctld to be ready
echo "Waiting for slurmctld to be ready..."
count=0
max_retries=10
while ! docker exec slurmctld scontrol ping; do
    echo "Waiting for slurmctld to become ready..."
    sleep 10
    ((count++))
    if [ "$count" -eq "$max_retries" ]; then
        echo "Failed to start slurmctld after $max_retries attempts."
        exit 1
    fi
done

# Register the cluster with the Slurm Database Daemon (slurmdbd)
echo "Registering the cluster with SlurmDBD..."
docker exec slurmctld ./register_cluster.sh

echo "Setup complete. JupyterHub is available at http://localhost:8000"
