#!/bin/bash

# This script installs Docker Compose on Ubuntu

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print an error message and exit
function error_exit {
    echo "$1" 1>&2
    exit 1
}

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    error_exit "Docker is not installed. Please install Docker first."
fi

# Install Docker Compose
echo "Installing Docker Compose..."

# Define the Docker Compose version to install
COMPOSE_VERSION="1.29.2"

# Download Docker Compose binary
sudo curl -L "https://github.com/docker/compose/releases/download/$COMPOSE_VERSION/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make the binary executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify the installation
if docker-compose --version &> /dev/null
then
    echo "Docker Compose installed successfully. Version: $(docker-compose --version)"
else
    error_exit "Docker Compose installation failed."
fi
