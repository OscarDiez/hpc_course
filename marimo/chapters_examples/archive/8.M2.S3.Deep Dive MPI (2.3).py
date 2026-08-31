import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 2.3 MPI
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to MPI in High-Performance Computing (HPC)

        High-Performance Computing (HPC) often involves running computational tasks that require massive parallelism across many processors. To achieve this, it's essential to use tools that can effectively manage communication between these processors.

        **Message Passing Interface (MPI)** is a standardized and portable message-passing system designed to function on a wide variety of parallel computing architectures. MPI is one of the cornerstones of parallel computing, particularly in distributed-memory systems, where each processor has its own memory and processors communicate by passing messages.

        In this lesson, we'll delve into the basics of MPI programming. You'll learn how to develop parallel applications that can efficiently communicate and share data across multiple processors. We'll explore MPI's core concepts through hands-on examples, starting with a simple yet powerful exercise known as the "ping-pong" example.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Overview of MPI Concepts

        Before diving into the code, it's important to understand some key concepts in MPI:

        - **Processes and Ranks**: In MPI, a process is an instance of a program running on a processor. Each process is assigned a unique identifier called a "rank." The rank is used to identify and communicate with other processes.
  
        - **Communicators**: A communicator defines a group of processes that can communicate with each other. The default communicator `MPI_COMM_WORLD` includes all the processes launched by the MPI program.

        - **Point-to-Point Communication**: This involves the direct sending and receiving of messages between two processes. MPI provides functions such as `MPI_Send` and `MPI_Recv` to facilitate this communication.

        - **Collective Communication**: This involves communication patterns where data is distributed among multiple processes or gathered from them. Examples include broadcast, scatter, and gather operations.

        These concepts form the foundation for writing parallel applications using MPI. Now, let's see how these concepts are applied in practice with the ping-pong example.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Setting Up MPI in Google Colab

        To run MPI programs, we first need to set up the MPI environment in Google Colab. We will use Open MPI, a popular implementation of the MPI standard. The first step is to install the necessary MPI libraries and tools.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Install MPI (mpich) in Google Colab
    !apt-get update -y
    !apt-get install -y mpich
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # The Ping-Pong Example

        The "ping-pong" program is a classic introductory example in MPI programming. It demonstrates how two processes can communicate by passing a message (or "ping-pong ball") back and forth. The program consists of two main steps:

        1. **Initialization**: Both processes initialize MPI, get their ranks, and determine who they will communicate with.

        2. **Message Passing**: The two processes take turns sending and receiving a message, incrementing a counter each time the message is passed. The process with rank 0 starts by sending the message to process 1. The message continues to be passed back and forth until a predefined count is reached.

        This example helps you understand the basic mechanics of point-to-point communication in MPI, including how messages are sent and received and how the rank of a process determines its role in the communication.

        """
    )
    return


@app.cell
def _():
    import os
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    return


@app.cell
def _():
    # Write the MPI ping pong example to a file
    mpi_code = """
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char** argv) {
        MPI_Init(NULL, NULL);

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // We are assuming at least 2 processes for this task
        if (world_size < 2) {
            fprintf(stderr, "World size must be greater than 1 for %s\\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int ping_pong_count = 0;
        int partner_rank = (world_rank + 1) % 2;
        while (ping_pong_count < 10) {
            if (world_rank == ping_pong_count % 2) {
                // Increment the ping pong count before you send it
                ping_pong_count++;
                MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
                printf("Process %d sent ping_pong_count %d to process %d\\n", world_rank, ping_pong_count, partner_rank);
            } else {
                MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Process %d received ping_pong_count %d from process %d\\n", world_rank, ping_pong_count, partner_rank);
            }
        }

        MPI_Finalize();
        return 0;
    }
    """

    with open('ping_pong.c', 'w') as f:
        f.write(mpi_code)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## MPI Ping Pong Example Explained

        The code provided implements a simple MPI (Message Passing Interface) "ping pong" program. This program demonstrates the basic concepts of point-to-point communication between two processes in an MPI environment. Below is a detailed explanation of the code.

        ### Code Overview

        1. **Initialization**:
           - `MPI_Init(NULL, NULL);`: Initializes the MPI environment. This must be called before any other MPI function. The `argc` and `argv` parameters allow MPI to take command-line arguments if needed.

        2. **Rank and Size**:
           - `MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);`: Determines the rank of the calling process in the communicator `MPI_COMM_WORLD`. The rank is the unique ID assigned to each process within the communicator, starting from 0.
           - `MPI_Comm_size(MPI_COMM_WORLD, &world_size);`: Determines the number of processes in the communicator `MPI_COMM_WORLD`.

        3. **Error Checking**:
           - The program assumes at least two processes for this example. If fewer than two processes are available, the program prints an error message and aborts using `MPI_Abort`.

        4. **Ping Pong Logic**:
           - The `ping_pong_count` variable tracks the number of messages sent back and forth.
           - `partner_rank = (world_rank + 1) % 2;`: Each process calculates the rank of its partner process. For two processes, rank 0's partner is rank 1, and rank 1's partner is rank 0.
           - The `while` loop continues until `ping_pong_count` reaches 10. The processes alternate sending and receiving the `ping_pong_count` value.
             - **Sending**: If the current process's rank matches the current `ping_pong_count % 2`, it increments the `ping_pong_count`, sends it to the partner process, and prints a message.
             - **Receiving**: If the current process's rank does not match `ping_pong_count % 2`, it waits to receive the `ping_pong_count` from the partner process and then prints a message.

        5. **Finalization**:
           - `MPI_Finalize();`: Cleans up the MPI environment. No MPI functions should be called after this.

        ### Key Concepts

        - **MPI_Comm_rank** and **MPI_Comm_size** are essential for identifying the process and determining the total number of processes involved.
        - **MPI_Send** and **MPI_Recv** are basic point-to-point communication functions, used here to send and receive the `ping_pong_count` variable between the two processes.
        - **Synchronization**: The processes are synchronized via alternating sends and receives, ensuring that the ping pong count is passed back and forth correctly.

        ### Example Output

        When you run this program with two processes, the output will look something like this:


        """
    )
    return


app._unparsable_cell(
    r"""
    # Compile the MPI program
    !mpicc -o ping_pong ping_pong.c
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Run the MPI program with 3 processes and allow running as root, with oversubscription
    !mpirun --oversubscribe -np 3 ./ping_pong
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Advanced MPI Example: Point-to-Point vs Collective Operations

        In this section, we will explore a more complex MPI example that illustrates the difference between point-to-point and collective operations. The program will use both types of MPI communication to demonstrate how they work and when each is appropriate.

        ### Code Overview

        The program performs the following tasks:
        1. **Initialization**:
           - As in the previous example, the program starts by initializing the MPI environment and determining the rank and size of the processes.

        2. **Data Distribution Using Point-to-Point Communication**:
           - Each process sends data to the next process in a ring-like fashion using `MPI_Send` and `MPI_Recv`.
           - This operation mimics a manual data distribution where each process explicitly sends and receives data to and from its neighbors.

        3. **Data Collection Using Collective Communication**:
           - All processes send their data to a root process using `MPI_Gather`, a collective operation that collects data from all processes and assembles it in the root process.

        4. **Broadcasting Data Using Collective Communication**:
           - The root process broadcasts data to all other processes using `MPI_Bcast`, another collective operation that efficiently distributes data from one process to all others.

        5. **Finalization**:
           - The program concludes by finalizing the MPI environment.

        ### Detailed Explanation

        1. **Initialization**:
           - `MPI_Init(NULL, NULL);`: Initializes the MPI environment.
           - `MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);`: Retrieves the rank (ID) of the calling process.
           - `MPI_Comm_size(MPI_COMM_WORLD, &world_size);`: Retrieves the total number of processes.

        2. **Point-to-Point Communication**:
           - **Sending Data**: Each process sends data to its neighbor using `MPI_Send`. For example, process 0 sends data to process 1, process 1 sends data to process 2, and so on. The last process sends data back to process 0, forming a ring.
           - **Receiving Data**: Simultaneously, each process receives data from its neighbor using `MPI_Recv`.
           - This operation is highly manual, as each process must explicitly specify the sender and receiver.

        3. **Collective Communication - Gathering Data**:
           - **MPI_Gather**: This operation is used to collect data from all processes and store it in a single root process. Each process sends its data to the root, where it is gathered into a single array or list.
           - Unlike point-to-point communication, `MPI_Gather` simplifies the process by automatically handling the collection of data from all processes.

        4. **Collective Communication - Broadcasting Data**:
           - **MPI_Bcast**: This operation broadcasts data from the root process to all other processes. It is an efficient way to distribute the same data to all processes in the communicator.
           - The root process sends its data once, and `MPI_Bcast` ensures that all processes receive it.

        5. **Finalization**:
           - `MPI_Finalize();`: Cleans up the MPI environment.

        ### Key Concepts

        - **Point-to-Point Communication**:
          - `MPI_Send` and `MPI_Recv` are used for direct communication between two processes.
          - This method is flexible but requires explicit management of senders and receivers, which can become complex in larger programs.

        - **Collective Communication**:
          - `MPI_Gather` and `MPI_Bcast` are collective operations that involve all processes in the communicator.
          - Collective operations are generally easier to use for common communication patterns, such as gathering data from all processes or broadcasting data to all processes.
          - Collective operations are often more efficient than equivalent point-to-point operations, especially on large numbers of processes.

        ### Example Output

        Running this program with four processes might produce output similar to the following:


        """
    )
    return


app._unparsable_cell(
    r"""
    # Save the MPI C code to a file
    mpi_code = \"\"\"
    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char** argv) {
        MPI_Init(NULL, NULL);

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Allocate some space for data
        int data = 100 + world_rank;  // Unique data for each process

        // Point-to-Point Communication: Ring Data Exchange
        int next_rank = (world_rank + 1) % world_size;
        int prev_rank = (world_rank - 1 + world_size) % world_size;
        int received_data;

        // Send data to the next process and receive data from the previous process
        MPI_Send(&data, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&received_data, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf(\"Process %d sent data %d to process %d and received data %d from process %d\\n\",
                world_rank, data, next_rank, received_data, prev_rank);

        // Collective Communication: Gather data at root
        int* gathered_data = NULL;
        if (world_rank == 0) {
            gathered_data = (int*)malloc(sizeof(int) * world_size);
        }
        MPI_Gather(&data, 1, MPI_INT, gathered_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            printf(\"Root process %d gathered data: \", world_rank);
            for (int i = 0; i < world_size; i++) {
                printf(\"%d \", gathered_data[i]);
            }
            printf(\"\\n\");
            free(gathered_data);
        }

        // Collective Communication: Broadcast data from root to all processes
        int broadcast_data = 500;
        if (world_rank == 0) {
            broadcast_data = 500;  // Root sets the data to be broadcasted
        }
        MPI_Bcast(&broadcast_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

        printf(\"Process %d received broadcast data: %d\\n\", world_rank, broadcast_data);

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Write the MPI code to a file
    with open('mpi_example.c', 'w') as f:
        f.write(mpi_code)

    # Compile the MPI C code
    !mpicc -o mpi_example mpi_example.c

    # Run the compiled MPI program with 4 processes
    !mpirun --oversubscribe -np 4 ./mpi_example
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of the MPI Program Output

        The output of the MPI program provides insight into how the data was communicated between processes using both point-to-point and collective operations. Let's break down the key parts of the output.

        ### Point-to-Point Communication (Ring Data Exchange)

        Each process sends its data to the next process in a circular manner (ring topology) and receives data from the previous process:

        - **Process 3 sent data 103 to process 0 and received data 102 from process 2**
          - Process 3 sends its data (103) to process 0.
          - Simultaneously, it receives data (102) from process 2.
  
        - **Process 0 sent data 100 to process 1 and received data 103 from process 3**
          - Process 0 sends its data (100) to process 1.
          - Simultaneously, it receives data (103) from process 3.
  
        - **Process 1 sent data 101 to process 2 and received data 100 from process 0**
          - Process 1 sends its data (101) to process 2.
          - Simultaneously, it receives data (100) from process 0.
  
        - **Process 2 sent data 102 to process 3 and received data 101 from process 1**
          - Process 2 sends its data (102) to process 3.
          - Simultaneously, it receives data (101) from process 1.

        This part of the output shows that each process successfully communicated with its neighbors in the ring. The data exchange is point-to-point, meaning each process explicitly sends and receives data from specific processes.

        ### Collective Communication - Gathering Data

        After the point-to-point communication, the program uses a collective operation, `MPI_Gather`, to collect data from all processes at the root process (process 0):

        - **Root process 0 gathered data: 100 101 102 103**
          - The root process (process 0) gathers data from all processes in the communicator.
          - The gathered data consists of the data from each process: 100 from process 0, 101 from process 1, 102 from process 2, and 103 from process 3.
  
        This output confirms that the `MPI_Gather` operation successfully collected data from all processes into the root process.

        ### Collective Communication - Broadcasting Data

        Finally, the program uses another collective operation, `MPI_Bcast`, to broadcast data from the root process (process 0) to all other processes:

        - **Process 0 received broadcast data: 500**
        - **Process 2 received broadcast data: 500**
        - **Process 1 received broadcast data: 500**
        - **Process 3 received broadcast data: 500**

        Here, the data value `500` is broadcasted by the root process (process 0) to all other processes. Each process receives this data and prints it, confirming that the broadcast was successful.

        ### Summary

        - **Point-to-Point Communication**: The data exchange between processes in a ring topology demonstrates how processes can communicate directly with each other using `MPI_Send` and `MPI_Recv`.
        - **Collective Communication - Gathering**: The `MPI_Gather` operation collects data from all processes and assembles it in the root process.
        - **Collective Communication - Broadcasting**: The `MPI_Bcast` operation efficiently distributes data from one process (the root) to all other processes.

        This output provides a clear example of both point-to-point and collective communication in an MPI program, showcasing how data can be exchanged and distributed among processes in a parallel computing environment.

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

