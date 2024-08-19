#!/bin/bash

# Define directories and files
BASE_DIR="$HOME/openhpc_cluster"
SHARED_DIR="$BASE_DIR/shared"
COMPOSE_FILE="$BASE_DIR/docker-compose.yml"
DOCKERFILE="$BASE_DIR/Dockerfile"
START_SCRIPT="$BASE_DIR/start.sh"

# Function to clean up previous Docker containers and networks
cleanup() {
    echo "Cleaning up old Docker containers and networks..."
    docker-compose -f "$COMPOSE_FILE" down --volumes --remove-orphans
    docker network prune -f
    docker container prune -f
    docker volume prune -f
}

# Create directories
mkdir -p "$SHARED_DIR"

# Create Dockerfile
cat <<EOF > "$DOCKERFILE"
# Dockerfile

FROM almalinux:9

# Install required packages
RUN dnf -y install epel-release && \\
    dnf config-manager --set-enabled crb && \\
    dnf -y install openssh-server openssh-clients sshpass slurm ohpc-base ohpc-warewulf ohpc-slurm-server munge && \\
    dnf clean all

# Configure SSH server
RUN mkdir /var/run/sshd && \\
    ssh-keygen -A && \\
    echo 'root:root' | chpasswd && \\
    sed -i 's/#PermitRootLogin yes/PermitRootLogin yes/' /etc/ssh/sshd_config && \\
    sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# Configure SLURM and other services
RUN mkdir -p /var/spool/slurm/ctld && \\
    mkdir -p /var/spool/slurm/d && \\
    chown slurm: /var/spool/slurm/ctld /var/spool/slurm/d

# Copy start script
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Expose SSH port
EXPOSE 22

# Start services directly
CMD ["/usr/local/bin/start.sh"]
EOF

# Create start.sh script
cat <<'EOF' > "$START_SCRIPT"
#!/bin/bash

# Start SSH daemon
/usr/sbin/sshd

# Start MUNGE
/usr/sbin/munged

# Start SLURM services
/usr/sbin/slurmctld
/usr/sbin/slurmd

# Keep the container running
tail -f /dev/null
EOF

# Create docker-compose.yml
cat <<EOF > "$COMPOSE_FILE"
version: '3.8'

services:
  master:
    build: .
    container_name: ohpc_master
    hostname: master
    privileged: true
    networks:
      ohpcnet:
        ipv4_address: 172.20.0.10
    volumes:
      - ./shared:/shared

  compute1:
    build: .
    container_name: compute1
    hostname: compute1
    privileged: true
    networks:
      ohpcnet:
        ipv4_address: 172.20.0.11
    volumes:
      - ./shared:/shared

  compute2:
    build: .
    container_name: compute2
    hostname: compute2
    privileged: true
    networks:
      ohpcnet:
        ipv4_address: 172.20.0.12
    volumes:
      - ./shared:/shared

  compute3:
    build: .
    container_name: compute3
    hostname: compute3
    privileged: true
    networks:
      ohpcnet:
        ipv4_address: 172.20.0.13
    volumes:
      - ./shared:/shared

networks:
  ohpcnet:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF

# Clean up existing setup
cleanup

# Build Docker images
echo "Building Docker images..."
docker-compose -f "$COMPOSE_FILE" build

# Start Docker Compose
echo "Starting Docker Compose..."
(cd "$BASE_DIR" && docker-compose up -d)

# Wait for containers to be ready
echo "Waiting for containers to start..."
sleep 20

# Function to install OpenHPC components on a node
install_openhpc() {
    local node=$1
    echo "Installing OpenHPC components on $node..."

    docker exec -it $node bash -c "
        dnf -y install epel-release &&
        dnf config-manager --set-enabled crb &&
        dnf -y install https://repos.openhpc.community/OpenHPC/3/EL_9/x86_64/ohpc-release-3-1.el9.x86_64.rpm &&
        dnf -y install ohpc-base ohpc-warewulf ohpc-slurm-server
    "

    echo "OpenHPC components installed on $node."
}

# Function to configure SSH keys for passwordless SSH
setup_ssh_keys() {
    local node=$1
    echo "Configuring SSH on $node..."

    docker exec -it $node bash -c "
        mkdir -p /root/.ssh &&
        ssh-keygen -t rsa -b 2048 -f /root/.ssh/id_rsa -q -N '' &&
        cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys &&
        chmod 600 /root/.ssh/authorized_keys
    "
}

# Install OpenHPC and configure SSH on master and compute nodes
install_openhpc ohpc_master
setup_ssh_keys ohpc_master

for i in {1..3}; do
    install_openhpc compute$i
    setup_ssh_keys compute$i
done

# Distribute the master's SSH key to all compute nodes
for i in {1..3}; do
    docker exec -it ohpc_master bash -c "
        sshpass -p 'root' ssh-copy-id -o StrictHostKeyChecking=no root@compute$i
    "
done

# Check SSH connectivity
echo "Checking SSH connectivity from master to compute nodes..."
for i in {1..3}; do
    docker exec -it ohpc_master bash -c "ssh -o BatchMode=yes -o ConnectTimeout=5 compute$i 'echo SSH to compute$i successful'"
    if [ $? -ne 0 ]; then
        echo "Failed to establish SSH connection to compute$i"
    else
        echo "SSH to compute$i successful"
    fi
done

# Configure SLURM on the master node
echo "Configuring SLURM on the master node..."

docker exec -it ohpc_master bash -c 'cat <<EOF > /etc/slurm/slurm.conf
# Basic SLURM configuration
ClusterName=openhpc
ControlMachine=master
SlurmUser=slurm
StateSaveLocation=/var/spool/slurm/ctld
SlurmdSpoolDir=/var/spool/slurm/d
AuthType=auth/munge
SchedulerType=sched/backfill
NodeName=compute[1-3] CPUs=1 State=UNKNOWN
PartitionName=debug Nodes=ALL Default=YES MaxTime=INFINITE State=UP
EOF'

# Start SLURM daemons manually without systemd
docker exec -it ohpc_master bash -c "/usr/sbin/slurmctld && /usr/sbin/munged"
for i in {1..3}; do
    docker exec -it compute$i bash -c "/usr/sbin/slurmd && /usr/sbin/munged"
done

# Check SLURM status
echo "Checking SLURM status..."
docker exec -it ohpc_master bash -c "sinfo"

echo "OpenHPC cluster setup complete."
