import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Distributed Filesystems in HPC

        ## Introduction

        In high-performance computing (HPC), distributed filesystems play a crucial role in providing scalable and efficient storage solutions across multiple nodes. One common approach is using Network File Systems (NFS) to allow file sharing between different machines in a cluster. Distributed filesystems ensure that users can access and share data seamlessly across the cluster nodes, which is essential for parallel processing and data-intensive workloads.

        In this tutorial, we will explore how NFS works in an HPC environment, how to mount a distributed filesystem, and how users can interact with such a system. You will learn:

        1. **What is NFS and its role in HPC clusters**.
        2. **How to set up and mount an NFS share**.
        3. **How to copy and manage files within a distributed filesystem**.
        4. **Best practices for using distributed filesystems in HPC**.

        This tutorial will be executed in Google Colab, simulating a distributed filesystem. For demonstration, we will install required software, create a simple NFS setup, and interact with the filesystem using basic commands.

        ---

        ## What is NFS?

        NFS, or Network File System, allows a computer to share directories and files with others over a network. It enables users on remote machines to interact with files on a server as if they were local, making it a popular choice in HPC for sharing large datasets across nodes.

        The main steps include:

        - Configuring the server to export a directory.
        - Configuring clients to mount that directory over the network.
        - Copying, accessing, and managing files as if they were local.

        ---

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Checking NFS Mounts in an HPC Cluster

        ## Introduction

        In an HPC (High-Performance Computing) environment, it's common to use a Network File System (NFS) to share directories across multiple nodes in the cluster. NFS allows users to access files and directories stored on a remote server as if they were on the local machine. Understanding how NFS works and how to check for mounted NFS directories is crucial for managing and using shared resources efficiently in an HPC environment.

        In this section, we will walk through the process of connecting to the cluster via SSH and checking the NFS mounts available on your system.

        ---

        ## Step 1: Connecting to the Cluster via SSH

        To interact with the HPC cluster and check for NFS mounts, you first need to connect to one of the cluster's nodes. You can do this using the `ssh` (Secure Shell) command, which allows you to securely log into a remote system.

        ### Example Command:
        ```bash
        ssh username@cluster_address
        ```
        Here, replace username with your cluster login username and cluster_address with the IP address or domain name of the cluster login node.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##Step 2: Checking for NFS Mounts
        Once logged into the cluster, you can use the mount command to check for mounted NFS directories. This will show you all the currently mounted filesystems, including any NFS shares.

        Example Command:

        ```bash
        mount | grep nfs
        ```
        This command filters the output of mount to show only NFS mounts. If NFS is being used in your cluster, you will see entries like the following:

        Example Output:
        ```bash
        10.0.1.6:/home on /home type nfs4 (rw,nosuid,noatime,seclabel,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,acregmax=3,acdirmin=3,acdirmax=3,hard,proto=tcp,timeo=600,retrans=2,sec=sys,clientaddr=10.0.1.4,local_lock=none,addr=10.0.1.6,_netdev,x-systemd.automount)
        10.0.1.6:/project on /project type nfs4 (rw,nosuid,noatime,seclabel,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,acregmax=3,acdirmin=3,acdirmax=3,hard,proto=tcp,timeo=600,retrans=2,sec=sys,clientaddr=10.0.1.4,local_lock=none,addr=10.0.1.6,_netdev,x-systemd.automount)
        10.0.1.6:/scratch on /scratch type nfs4 (rw,nosuid,noatime,seclabel,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,acregmax=3,acdirmin=3,acdirmax=3,hard,proto=tcp,timeo=600,retrans=2,sec=sys,clientaddr=10.0.1.4,local_lock=none,addr=10.0.1.6,_netdev,x-systemd.automount)
        ```

        ###Step 3: Understanding the Output
        NFS Server: The IP address or hostname of the NFS server is the first part of each line, such as 10.0.1.6:/home.

        Mount Point: The location on the local node where the NFS share is mounted, for example /home, /project, or /scratch.

        NFS Version: The version of NFS being used is indicated, such as nfs4.

        Mount Options: These are the options used for the NFS mount, which control read/
        write permissions, timeouts, and other settings. For example, rw (read/write), noatime (no access time update), vers=4.2 (NFS version 4.2), proto=tcp (using TCP), and many others.

        Example Commands:
        List all mounted filesystems:

        ```bash
        mount
        ```
        This command shows all mounted filesystems, including NFS, local disks, and other mounts.

        View only NFS mounts:

        ```bash
        mount | grep nfs
        ```
        Filters the output to show only NFS mounts.

        Check disk usage of NFS mounts:

        ```bash
        df -hT | grep nfs
        ```
        Displays disk space usage for mounted NFS directories, including the size, used space, and free space.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Parallel I/O with MPI-IO and HDF5

        In high-performance computing (HPC) environments, efficient I/O operations are critical to overall performance. **Parallel I/O** refers to the ability of multiple processes to read and write data simultaneously to a shared file. This is particularly important for scientific applications that work with large datasets.

        In this example, we will demonstrate two parallel I/O methods using:
        1. **MPI-IO**: A low-level interface provided by the MPI library that allows parallel read/write operations to a shared file.
        2. **HDF5**: A high-level data format designed for large-scale data management, which supports parallel I/O.

        Both methods will be executed on an HPC cluster using multiple processes, and data will be written in parallel to a shared file.

        ### Objectives:
        - Learn how to use MPI-IO to write data in parallel from multiple processes.
        - Understand the benefits of using HDF5 for parallel data management.

        ### Requirements:
        - The cluster needs to have **MPI**, **HDF5**, and **parallel file systems** like Lustre installed.
        - This example assumes a multi-node setup with JupyterLab access on the HPC cluster.


        ---
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Parallel I/O with MPI-IO and HDF5 (Cluster one)
        ### Running Parallel HDF5 Example on the Cluster

        This guide will walk you through connecting to the cluster, setting up the environment, compiling an HDF5 example program, and running it.

        ### Step 1: Connect to the Cluster via SSH

        To connect to the cluster, open a terminal and use the `ssh` command. Replace `username` and `cluster_address` with the appropriate values provided by your system administrator.

        ```bash
        ssh username@cluster_address

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Once connected, you will be in your home directory on the cluster.

        ###Step 2: Load the Necessary Modules
        Before compiling and running your HDF5 example, you need to load the required modules for the environment. Run the following commands:

        ```bash
        module load gcc/9.3.0
        module load openmpi/4.0.3
        module load hdf5-mpi/1.12.1
        ```
        You can check the currently loaded modules using:

        ```bash
        module list
        ```
        Make sure that the hdf5-mpi module is listed.

        ###Step 3: Create the C File with Nano
        Now, we will create the HDF5 C program. Use nano to create a new file named parallel_hdf5_example.c.

        ```bash
        nano parallel_hdf5_example.c
        ```
        Copy and paste the following sample HDF5 program into the file:

        ```c
        #include <mpi.h>
        #include <hdf5.h>
        #include <stdio.h>

        int main(int argc, char **argv) {
            MPI_Init(&argc, &argv);

            hid_t file_id; /* File identifier */
            herr_t status;

            /* Create a new file collectively using default properties. */
            file_id = H5Fcreate("test_parallel.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            /* Close the file */
            status = H5Fclose(file_id);

            MPI_Finalize();
            return 0;
        }
        ```

        To save and exit in nano, press CTRL + O to save, then CTRL + X to exit.

        ###Step 4: Compile the HDF5 Program
        Now that the file is created, compile it using mpicc (the MPI-enabled C compiler) with the following command:

        ```bash
        mpicc parallel_hdf5_example.c -o parallel_hdf5_example -lhdf5 -lhdf5_hl
        ```

        This command compiles the program and links the necessary HDF5 libraries.

        If you receive any errors, make sure the hdf5-mpi module is loaded, and the correct paths are set.

        ###Step 5: Run the Program
        Once the program is compiled, run it using mpirun:

        ```bash
        mpirun -np 2 ./parallel_hdf5_example
        ```
        The -np 2 flag tells mpirun to run the program on 2 processes. You can adjust the number of processes as needed. If it fails you can try to run it with -oversubscribe option

        ###Step 6: Verify the Output
        If everything works correctly, a file named test_parallel.h5 should be created in your current directory. To verify it, list the files in the directory:

        ```bash
        ls
        ```
        You should see test_parallel.h5 in the output. If it's there, the program ran successfully.

        ###Step 7: Clean Up (Optional)
        To remove the generated files, you can run:

        ```bash
        rm parallel_hdf5_example test_parallel.h5
        ```
        This removes both the compiled binary and the generated HDF5 file.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Guidelines for Designing I/O Systems in HPC

        ## Introduction
        In this practice, we will explore the key guidelines for designing efficient I/O systems in high-performance computing (HPC) environments. Efficient I/O is crucial for improving performance in HPC systems, particularly when dealing with massive datasets that are common in scientific and engineering applications.

        We will focus on the following concepts:
        - **Optimizing Data Locality & Access Patterns**: Placing data close to compute resources to reduce latency.
        - **Leveraging Parallel I/O Techniques**: Using libraries like MPI-IO and NetCDF to perform parallel file operations.
        - **Implementing I/O Scheduling & Load Balancing**: Distributing I/O operations across resources to balance the load.
        - **Incorporating Resilience & Fault Tolerance**: Implementing mechanisms to handle I/O failures and ensure data integrity.

        The goal of this exercise is to learn about these principles through hands-on experience. We will write and compile C programs that demonstrate these concepts and execute them in a Google Colab environment.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Install necessary tools in Google Colab
    !apt-get install mpich
    !apt-get install gcc
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Create the C file to demonstrate parallel I/O using MPI-IO with enhanced logging

    code = \"\"\"
    #include <stdio.h>
    #include <mpi.h>

    #define FILENAME \"testfile.bin\"

    int main(int argc, char **argv) {
        int rank, size;
        MPI_File file;
        MPI_Status status;
        MPI_Init(&argc, &argv);  // Initialize MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process
        MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the size of the communicator

        // Inform the user about the number of processes
        if (rank == 0) {
            printf(\"Running Parallel I/O with %d processes...\\n\", size);
        }

        // Open the file for parallel I/O
        MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

        // Each process writes its rank multiplied by 10 to a different part of the file
        int buf = rank * 10;
        MPI_Offset offset = rank * sizeof(int);

        // Print logging information before writing
        printf(\"Process %d writing value %d at offset %lld\\n\", rank, buf, (long long)offset);

        // Perform the write operation
        MPI_File_write_at(file, offset, &buf, 1, MPI_INT, &status);

        // Print logging information after writing
        printf(\"Process %d finished writing\\n\", rank);

        // Close the file
        MPI_File_close(&file);

        MPI_Finalize();  // Finalize MPI
        return 0;
    }
    \"\"\"

    # Write the C code to a file
    with open(\"parallel_io.c\", \"w\") as file:
        file.write(code)

    # Compile the C code
    !mpicc -o parallel_io parallel_io.c

    # Run the parallel I/O program with 4 processes, allowing root execution and printing details
    !mpirun --allow-run-as-root -np 4 -oversubscribe ./parallel_io
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of the Code

        In this section, we will go through the key components of the C code used to demonstrate parallel I/O in an HPC environment:

        ### MPI Initialization
        ```c
        MPI_Init(&argc, &argv);
        ```

        This initializes the MPI environment, which is required for any parallel operations using MPI. Every MPI program must call this at the beginning.

        Opening the File for Parallel I/O
        ```c
        MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        ```

        Here, we open a file (testfile.bin) in parallel using MPI_File_open. The flag MPI_MODE_CREATE | MPI_MODE_WRONLY indicates that the file should be created if it doesn't exist and opened in write-only mode.

        Writing Data by Each Process

        ```c
        MPI_File_write_at(file, rank * sizeof(int), &buf, 1, MPI_INT, &status);
        ```

        Each process writes its data at a different offset in the file. The offset is determined by rank * sizeof(int), so that each process writes its data to a unique location. This demonstrates parallel I/O, where multiple processes can write to the same file simultaneously without conflict.

        Finalizing MPI

        ```c
        MPI_Finalize();
        ```
        Once all the operations are complete, we call MPI_Finalize to clean up the MPI environment.

        Running the Program
        The program is executed using mpirun with 4 processes, as shown below:

        ```bash
        mpirun -np 4 ./parallel_io
        ```
        This means that 4 processes will be used to write their respective data to the file concurrently. After the program runs, each process will have written its rank multiplied by 10 to the file at its corresponding offset.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Two-Phase Commit Protocol in HPC

        ## Introduction
        The two-phase commit protocol is essential in distributed systems like High-Performance Computing (HPC), where ensuring atomicity and consistency of transactions across multiple nodes is critical. This protocol is used in scenarios such as checkpointing—saving the current state of a computation across many nodes—ensuring that the checkpoint operation is either committed or aborted simultaneously on all nodes.

        ### Why is it important?
        In distributed HPC environments, if nodes become out of sync during operations like checkpointing, it can lead to incorrect or inconsistent computation results. The two-phase commit ensures that either all nodes commit a transaction (e.g., saving a checkpoint) or all nodes abort it if any node fails to do so, preserving data integrity across the system.

        In this exercise, we will simulate a simplified version of the two-phase commit protocol using Python to represent a distributed checkpoint operation.

        ## How It Works
        The protocol works in two phases:
        1. **Preparation Phase**: Each node prepares to perform the checkpoint, responding with either "YES" (ready to commit) or "NO" (cannot commit).
        2. **Commit/Abort Phase**: If all nodes vote "YES," the coordinator instructs them to commit the checkpoint. If any node votes "NO," the coordinator instructs all nodes to abort the checkpoint.

        We will simulate this using a coordinator node and multiple worker nodes.

        """
    )
    return


@app.cell
def _():
    # Simulating Two-Phase Commit Protocol in HPC for Checkpointing

    import random

    class Node:
        def __init__(self, name):
            self.name = name
            self.checkpoint_ready = False

        def prepare_checkpoint(self, data):
            """ Simulate preparation for checkpointing """
            # Randomly decide if the node can save the checkpoint
            self.checkpoint_ready = random.choice([True, False])
            print(f"{self.name}: Preparing checkpoint... {'YES' if self.checkpoint_ready else 'NO'}")
            return "YES" if self.checkpoint_ready else "NO"

        def commit_checkpoint(self, data):
            """ Commit checkpoint if ready """
            if self.checkpoint_ready:
                print(f"{self.name}: Committing checkpoint...")
            else:
                print(f"{self.name}: Cannot commit, not ready!")

        def abort_checkpoint(self):
            """ Abort checkpoint operation """
            print(f"{self.name}: Aborting checkpoint...")

    class Coordinator:
        def __init__(self, nodes):
            self.nodes = nodes

        def perform_checkpoint(self, checkpoint_data):
            print("Coordinator: Initiating checkpoint...")
            votes = []
            for node in self.nodes:
                vote = node.prepare_checkpoint(checkpoint_data)
                votes.append(vote)

            # Check if all nodes voted "YES"
            if all(vote == "YES" for vote in votes):
                print("Coordinator: All nodes voted YES. Committing checkpoint...")
                for node in self.nodes:
                    node.commit_checkpoint(checkpoint_data)
            else:
                print("Coordinator: Some nodes voted NO. Aborting checkpoint...")
                for node in self.nodes:
                    node.abort_checkpoint()

    # Simulating the two-phase commit with nodes
    nodes = [Node(f"Node {i+1}") for i in range(4)]  # Create 4 nodes
    coordinator = Coordinator(nodes)

    # Perform checkpoint
    checkpoint_data = "checkpoint_data"
    coordinator.perform_checkpoint(checkpoint_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of the Two-Phase Commit Code

        In this simulation, we modeled a simplified version of the two-phase commit protocol using a coordinator and multiple nodes. Each node decides whether it is ready to commit a checkpoint based on a random decision.

        ### Key Components:
        1. **Node Class**:
            - `prepare_checkpoint`: Each node prepares for the checkpoint and returns "YES" if ready, otherwise "NO".
            - `commit_checkpoint`: If the node is ready, it commits the checkpoint.
            - `abort_checkpoint`: If the node is not ready, it aborts the checkpoint operation.
  
        2. **Coordinator Class**:
            - `perform_checkpoint`: The coordinator manages the entire operation. It asks each node to prepare for the checkpoint and collects their votes. If all nodes vote "YES," it instructs them to commit the checkpoint. If any node votes "NO," it aborts the checkpoint on all nodes.

        ### Output:
        When you run the program, you will see messages from the coordinator and nodes about whether they are ready to commit or abort the checkpoint. This simulates how a two-phase commit would work in an actual HPC system where the consistency of checkpoints across nodes is crucial.

        The two-phase commit protocol ensures that either all nodes commit or all abort, preventing inconsistent states across the cluster.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Consensus Algorithms in HPC

        ## Introduction
        In distributed systems, such as High-Performance Computing (HPC) environments, it is often necessary for nodes to reach a consensus on shared states or configurations. Consensus algorithms like Paxos and Raft are used to ensure agreement across nodes, even in the presence of failures.

        ### Why is Consensus Important in HPC?
        Consensus algorithms are vital in HPC for tasks like distributed job scheduling, maintaining consistency of shared state, or synchronizing tasks across many nodes. Without proper coordination, the system could face inconsistencies, leading to resource contention or incorrect computations.

        In this example, we will simulate a simplified version of a consensus algorithm using a class-based Python model to illustrate how nodes in an HPC system might reach consensus to elect a leader that coordinates tasks across a cluster.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the Python code to a file
    consensus_code = \"\"\"
    import time

    class HPCNode:
        def __init__(self, node_id):
            self.node_id = node_id
            self.state = \"follower\"
            self.current_term = 0
            self.voted_for = None

        def start_election(self, cluster_nodes):
            ''' Initiates leader election '''
            self.state = \"candidate\"
            self.current_term += 1
            print(f\"Node {self.node_id} is starting an election (Term {self.current_term})...\")

            votes = 1  # Vote for self

            # Request votes from other nodes
            for node in cluster_nodes:
                if node != self and node.vote_request(self.current_term, self.node_id):
                    votes += 1

            # Majority wins
            if votes > len(cluster_nodes) // 2:
                self.state = \"leader\"
                print(f\"Node {self.node_id} is elected leader with {votes} votes!\")
                self.coordinate_tasks(cluster_nodes)
            else:
                print(f\"Node {self.node_id} failed to become leader.\")

        def vote_request(self, term, candidate_id):
            ''' Respond to a vote request '''
            if term > self.current_term:
                self.current_term = term
                self.voted_for = candidate_id
                print(f\"Node {self.node_id} votes for Node {candidate_id}\")
                return True
            return False

        def coordinate_tasks(self, cluster_nodes):
            ''' Leader coordinates tasks across the cluster '''
            print(f\"Node {self.node_id} is now coordinating tasks...\")
            for i in range(3):  # Simulate coordinating 3 tasks
                print(f\"Node {self.node_id}: Coordinating task {i+1}\")
                time.sleep(1)  # Simulate time taken to coordinate tasks

    # Create nodes and simulate an election
    nodes = [HPCNode(i) for i in range(5)]  # Create 5 nodes in the cluster
    random_node = random.choice(nodes)  # Randomly select a node to start an election
    random_node.start_election(nodes)
    \"\"\"

    # Save the code to a Python file
    with open(\"consensus_algorithm.py\", \"w\") as f:
        f.write(consensus_code)

    # Run the Python file
    !python3 consensus_algorithm.py
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of the Consensus Algorithm Code

        In this simulation, we implemented a simple consensus algorithm for leader election in an HPC environment, similar to the Raft algorithm. Nodes elect a leader to coordinate tasks.

        ### Key Components:
        1. **HPCNode Class**:
            - `start_election`: The node becomes a candidate and requests votes from other nodes. If it receives a majority of votes, it becomes the leader.
            - `vote_request`: Nodes respond to vote requests and grant their vote to the candidate with the highest term.
            - `coordinate_tasks`: Once a node is elected leader, it coordinates tasks across the cluster.

        2. **Leader Election**:
            - A node randomly initiates the election. If the node receives votes from a majority of nodes, it becomes the leader and begins coordinating tasks.
            - If the node fails to receive a majority, the election fails, and the process can be restarted.

        ### Output:
        When you run the program, you will see messages indicating the election process, including which nodes voted for the candidate and whether a leader was successfully elected. Once elected, the leader will begin coordinating tasks.

        ### Importance in HPC:
        Consensus algorithms ensure that tasks are coordinated effectively across distributed nodes in an HPC environment. This is critical for job scheduling, resource management, and maintaining consistency across the system.

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

