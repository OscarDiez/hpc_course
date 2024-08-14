#!/bin/bash

# Start the OpenHPC Docker cluster
echo "Starting the OpenHPC Docker cluster..."
docker-compose up -d

# Check the status of the containers
echo "Checking Docker container status..."
containers=("ohpc_master" "compute1" "compute2" "compute3")
for container in "${containers[@]}"; do
  if docker ps | grep -q "$container"; then
    echo "Container $container is running"
  else
    echo "ERROR: Container $container is not running. Please check Docker logs."
    exit 1
  fi
done

# Define nodes and master
nodes=("compute1" "compute2" "compute3")
master="ohpc_master"

# Check SSH connectivity
echo "Checking node connectivity..."
for node in "${nodes[@]}" "$master"; do
  echo "Testing SSH connection to $node..."
  docker exec -it $master ssh -o BatchMode=yes -o ConnectTimeout=5 $node 'echo "SSH to '$node' successful"'
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to connect to $node via SSH. Please check network and SSH configuration."
    exit 1
  else
    echo "Successfully connected to $node"
  fi
done

# Check if SLURM daemons are running
echo "Checking SLURM daemons..."
services=("slurmctld" "munge")
for node in $master; do
  for service in "${services[@]}"; do
    docker exec -it $master ssh $node systemctl is-active $service >/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "ERROR: $service is not running on $node"
      exit 1
    else
      echo "$service is running on $node"
    fi
  done
done

# Check SLURM on compute nodes
for node in "${nodes[@]}"; do
  docker exec -it $master ssh $node systemctl is-active slurmd >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "ERROR: slurmd is not running on $node"
    exit 1
  else
    echo "slurmd is running on $node"
  fi
done

# Check SLURM node availability
echo "Checking SLURM node availability..."
docker exec -it $master sinfo -N -h | awk '{print $1, $4}' | while read -r node state; do
  echo "Node $node is in state $state"
  if [[ "$state" != "idle" && "$state" != "alloc" ]]; then
    echo "ERROR: Node $node is not in an expected state: $state"
    exit 1
  fi
done

# Compile and run MPI job
echo "Compiling and submitting MPI job..."
docker exec -it $master bash -c 'echo -e "#include <mpi.h>\n#include <stdio.h>\nint main(int argc, char** argv) { MPI_Init(&argc, &argv); int world_size, world_rank; MPI_Comm_size(MPI_COMM_WORLD, &world_size); MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); printf("Hello from rank %d out of %d processors\\n", world_rank, world_size); MPI_Finalize(); return 0; }" > mpi_test.c'
docker exec -it $master mpicc mpi_test.c -o mpi_test

echo "Submitting the MPI test job..."
docker exec -it $master bash -c 'echo -e "#!/bin/bash\n#SBATCH --job-name=mpi_test\n#SBATCH --nodes=2\n#SBATCH --ntasks-per-node=2\n#SBATCH --time=00:01:00\nsrun ./mpi_test" > mpi_test.slurm'
docker exec -it $master sbatch mpi_test.slurm

echo "Waiting for MPI job to complete..."
sleep 60

# Check MPI job output
echo "Checking MPI job output..."
mpi_output=$(docker exec -it $master grep -q "Hello from rank" mpi_test_output.txt)
if [ $? -eq 0 ]; then
  echo "MPI test job completed successfully"
else
  echo "ERROR: MPI test job failed"
  exit 1
fi

echo "All checks passed. OpenHPC setup is working correctly."
