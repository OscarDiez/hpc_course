#!/bin/bash

# Ensures the script is run as root or with sudo privileges
if [ "$(id -u)" != "0" ]; then
    echo "This script must be run as root" 1>&2
    exit 1
fi

echo "Updating package database..."
apt-get update

echo "Installing required packages..."
apt-get install -y \
    ca-certificates \
    curl \
    software-properties-common \
    gnupg \
    lsb-release

echo "Adding Docker's official GPG key..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "Setting up the stable Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "Updating the package database again..."
apt-get update

echo "Installing Docker Engine, Docker CLI, and Containerd..."
apt-get install -y docker-ce docker-ce-cli containerd.io

echo "Adding the current user to the Docker group..."
usermod -aG docker $SUDO_USER

echo "Applying group changes..."
newgrp docker

echo "Enabling Docker to start on boot..."
systemctl enable docker

echo "Docker installation complete. Docker version:"
docker --version
