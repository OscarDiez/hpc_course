#!/bin/bash

# Update package index
sudo apt update

# Install necessary packages
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository to APT sources
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Update package index again with Docker packages
sudo apt update

# Install Docker CE
sudo apt install -y docker-ce

# Enable Docker service
sudo systemctl enable docker

# Start Docker service
sudo systemctl start docker

# Add the current user to the Docker group to allow running Docker without sudo
sudo usermod -aG docker ${USER}

# Download Docker Compose (specify the version you need, replace '1.29.2' with the latest if necessary)
DOCKER_COMPOSE_VERSION="1.29.2"
sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Apply executable permissions to the Docker Compose binary
sudo chmod +x /usr/local/bin/docker-compose

# Test Docker and Docker Compose installations
echo "Testing Docker installation..."
docker --version

echo "Testing Docker Compose installation..."
docker-compose --version

echo "Installation complete! Please log out and log back in to apply group membership changes."

