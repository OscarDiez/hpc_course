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


@app.cell
def _():
    import os
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    return (os,)


@app.cell
def _():
    import subprocess

    # 1. Write the C code to a file
    c_code = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <mpi.h>
    #include <string.h>

    int main(int argc,char **argv)
    {
        int rank, size;
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_size(MPI_COMM_WORLD,&size);

        int message[2];    // buffer for sending and receiving messages
        int dest, src;     // destination and source process variables
        int tag = 0;
        MPI_Status status;

        // This example has to be run on more than one process
        if (size == 1) {
            printf("This example requires >1 process to execute\\n");
            MPI_Finalize();
            exit(0);
        }

        if (rank != 0) {
            // If not rank 0, send message to rank 0
            message[0] = rank;
            message[1] = size;
            dest = 0;  // send all messages to rank 0
            MPI_Send(message, 2, MPI_INT, dest, tag, MPI_COMM_WORLD);
        } else {
            // If rank 0, receive messages from everybody else
            for (src = 1; src < size; src++) {
                MPI_Recv(message, 2, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                // prints message just received. Notice it will print in rank
                // order since the loop is in rank order.
                printf("Hello from process %d of %d\\n", message[0], message[1]);
            }
        }

        MPI_Finalize();
        return 0;
    }
    """

    # Write the C code to a file
    with open("mpi_program.c", "w") as c_file:
        c_file.write(c_code)

    print("C program written to 'mpi_program.c'")

    # 2. Compile the C code using mpicc
    compile_command = ["mpicc", "-o", "mpi_program", "mpi_program.c"]
    subprocess.run(compile_command, check=True)
    print("C program compiled successfully")

    # 3. Run the compiled program with 4 nodes and allow oversubscription
    run_command = ["mpirun", "--oversubscribe", "-np", "4", "./mpi_program"]
    try:
        result = subprocess.run(run_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())  # Output from the command
    except subprocess.CalledProcessError as e:
        print("Error occurred while running MPI program:", e.stderr.decode())  # Print error output
    return


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
def _(os):
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    return


@app.cell
def _():
    # Step 1: Writing the MPI ping-pong example to a file
    mpi_code = """
    #include <mpi.h>
    #include <stdio.h>
    #include <unistd.h> // For sleep

    int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Ensure that there are at least 2 processes
        if (world_size < 2) {
            fprintf(stderr, "World size must be greater than 1 for %s\\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int ping_pong_count = 0;
        int partner_rank = (world_rank + 1) % 2;
        const int MAX_COUNT = 10;  // Reduced number of ping-pong iterations

        while (ping_pong_count < MAX_COUNT) {
            if (world_rank == ping_pong_count % 2) {
                // Increment the count before sending
                ping_pong_count++;
                MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
                printf("Process %d sent ping_pong_count %d to process %d\\n", world_rank, ping_pong_count, partner_rank);
                fflush(stdout); // Ensure output is flushed
                sleep(1);  // Small delay to avoid flooding
            } else {
                MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Process %d received ping_pong_count %d from process %d\\n", world_rank, ping_pong_count, partner_rank);
                fflush(stdout); // Ensure output is flushed
            }
        }

        MPI_Finalize();
        return 0;
    }
    """

    # Save the MPI code to a file
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
    # Run the MPI program with 2 processes and allow running as root, with oversubscription
    !mpirun --oversubscribe -np 2 ./ping_pong
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # MPI Dot Product Example

        In this example, we compute the dot product of two vectors distributed across multiple processes using MPI. Each process computes a portion of the dot product (local dot product), and the final result is obtained by reducing (summing) all the local dot products at the root process.

        ### Key Concepts
        1. **Process Distribution**: Each process holds a portion of the vectors `a` and `b`, and computes the dot product of its portion. The size of each local vector is constant across all processes.
        2. **MPI_Reduce**: This MPI function is used to collect the partial dot products from all processes and sum them at the root process.

        ### Steps in the Code
        1. **Vector Initialization**: Each process initializes its portion of the vectors `a` and `b` based on its rank.
        2. **Partial Dot Product Calculation**: Each process computes the dot product for its portion of the vectors.
        3. **Reduction**: The partial dot products from all processes are reduced (summed) to compute the final dot product at the root process.
        4. **Final Output**: The root process prints the final dot product.

        ### Code Walkthrough
        - **MPI Initialization**: We initialize MPI using `MPI_Init()`, and each process determines its rank and the total number of processes.
        - **Vector Initialization**: Each process allocates memory for its local portion of the vectors `a` and `b` and initializes them based on its rank.
        - **Partial Dot Product**: Each process calculates the dot product of its local vectors.
        - **MPI_Reduce**: The partial dot products are summed up at the root process using `MPI_Reduce()`.
        - **Final Output**: The root process prints the final dot product.

        ### Task:
        Run this code using 4 processes and check the result.

        ### MPI Code Compilation and Execution in Jupyter
        The following code will write the MPI program to a file, compile it using `mpicc`, and run it with oversubscription in case you are running more processes than CPU cores.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 1: Writing the corrected MPI code to a file
    mpi_code = \"\"\"
    #include <stdlib.h>
    #include <stdio.h>
    #include <mpi.h>

    int main(int argc, char **argv) {
        MPI_Init(&argc, &argv);
        int rank, p, i, root = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);

        // Make the local vector size constant
        int local_vector_size = 100;

        // Compute the global vector size
        int n = p * local_vector_size;

        // Initialize the vectors
        double *a, *b;
        a = (double *) malloc(local_vector_size * sizeof(double));
        b = (double *) malloc(local_vector_size * sizeof(double));
        for (i = 0; i < local_vector_size; i++) {
            a[i] = 3.14 * rank;
            b[i] = 6.67 * rank;
        }

        // Compute the local dot product
        double partial_sum = 0.0;
        for (i = 0; i < local_vector_size; i++) {
            partial_sum += a[i] * b[i];
        }

        double sum = 0;
        MPI_Reduce(&partial_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

        if (rank == root) {
            // Corrected printf statement
            printf(\"The dot product is %g\\n\", sum);
        }

        free(a);
        free(b);
        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Step 2: Save the corrected MPI code to a file
    with open('mpi_dot_product.c', 'w') as f:
        f.write(mpi_code)

    # Step 3: Compile the MPI program using mpicc
    !mpicc -o mpi_dot_product mpi_dot_product.c

    # Step 4: Run the compiled MPI program with 4 processes and oversubscription
    !mpirun --oversubscribe -np 4 ./mpi_dot_product
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Modify the MPI Dot Product Program

        In this exercise, you will modify the existing MPI program to better understand how the `MPI_Reduce` function works and how to handle larger vectors efficiently.

        ### Task 1: Experiment with Larger Vector Sizes
        Currently, the local vector size is set to 100 for each process. Modify the program so that:
        1. The local vector size is increased to **1000** elements for each process.
        2. Observe the effect this change has on the **dot product** calculation. Does the result scale as expected?

        ### Task 2: Use Different Operations in `MPI_Reduce`
        Currently, `MPI_Reduce` is used to sum the partial dot products from each process. Modify the program so that:
        1. Instead of summing, you use **`MPI_MAX`** to find the maximum dot product contribution from the processes.
        2. Print the result using the `MPI_MAX` operation to see how the values from different processes contribute to the final result.

        ### Hints:
        - You can change the **reduce operation** in `MPI_Reduce` by replacing `MPI_SUM` with `MPI_MAX`.
        - Use larger vectors to understand the impact of data size on performance.
        - Make sure to print and compare the results with both **sum** and **maximum** reductions.

        After completing the tasks, run the program with different vector sizes and observe how the results and performance change.

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


app._unparsable_cell(
    r"""
    # MPI Collective Operations: A Hands-On Example

    In this section, we will explore some of the most popular MPI collective operations, which allow processes to communicate in different patterns. We will use a small vector and perform the following collective operations:
    - **MPI_Bcast**: Broadcasts data from one process (root) to all other processes.
    - **MPI_Scatter**: Divides the data into chunks and distributes them across multiple processes.
    - **MPI_Gather**: Gathers data from all processes and combines it at the root process.
    - **MPI_Reduce**: Reduces values from all processes (e.g., summing them) and stores the result at the root process.

    ### Key Collective Operations:
    1. **MPI_Bcast**:
       - The root process broadcasts a vector to all other processes.
       - All processes receive the same vector from the root.

    2. **MPI_Scatter**:
       - A vector is divided into equal parts, and each process receives one part (a chunk).

    3. **MPI_Gather**:
       - Each process contributes a small vector (chunk), and the root process gathers these chunks to form the original vector.

    4. **MPI_Reduce**:
       - Each process computes a local sum, and the root process reduces these sums (e.g., summing them all) to compute the global sum.

    ### Example:
    We will use a small vector of 8 elements and visualize how each operation modifies the data across 4 processes. The root process will print the result for each operation, allowing you to see the differences between broadcasting, scattering, gathering, and reducing.

    ### Task:
    Run the provided code with 4 processes and observe the data exchanges in each collective operation.

    Let's now compile and run the code using MPI.
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Step 1: Writing the MPI code to a file
    mpi_code = \"\"\"
    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int root = 0;
        int vector_size = 8;
        int local_vector_size = vector_size / size;
        int i;

        // Define a vector to be used in the operations
        int *vector = NULL;
        if (rank == root) {
            vector = (int *)malloc(vector_size * sizeof(int));
            for (i = 0; i < vector_size; i++) {
                vector[i] = i + 1;
            }
            printf(\"Root process initial vector: \");
            for (i = 0; i < vector_size; i++) {
                printf(\"%d \", vector[i]);
            }
            printf(\"\\n\");
        }

        // Broadcast: Send the vector from root to all processes
        int recv_vector_bcast[vector_size];
        MPI_Bcast(recv_vector_bcast, vector_size, MPI_INT, root, MPI_COMM_WORLD);
        printf(\"Process %d received broadcasted vector: \", rank);
        for (i = 0; i < vector_size; i++) {
            printf(\"%d \", recv_vector_bcast[i]);
        }
        printf(\"\\n\");

        // Scatter: Send chunks of the vector to each process
        int local_vector[local_vector_size];
        MPI_Scatter(vector, local_vector_size, MPI_INT, local_vector, local_vector_size, MPI_INT, root, MPI_COMM_WORLD);
        printf(\"Process %d received scattered vector: \", rank);
        for (i = 0; i < local_vector_size; i++) {
            printf(\"%d \", local_vector[i]);
        }
        printf(\"\\n\");

        // Gather: Collect local vectors from all processes to the root process
        int *gathered_vector = NULL;
        if (rank == root) {
            gathered_vector = (int *)malloc(vector_size * sizeof(int));
        }
        MPI_Gather(local_vector, local_vector_size, MPI_INT, gathered_vector, local_vector_size, MPI_INT, root, MPI_COMM_WORLD);
        if (rank == root) {
            printf(\"Root process gathered vector: \");
            for (i = 0; i < vector_size; i++) {
                printf(\"%d \", gathered_vector[i]);
            }
            printf(\"\\n\");
        }

        // Reduce: Compute the sum of the local vectors and reduce at root
        int local_sum = 0;
        for (i = 0; i < local_vector_size; i++) {
            local_sum += local_vector[i];
        }
        int global_sum = 0;
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
        if (rank == root) {
            printf(\"Global sum after reduction: %d\\n\", global_sum);
        }

        // Clean up
        if (rank == root) {
            free(vector);
            free(gathered_vector);
        }

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Step 2: Save the MPI code to a file
    with open('mpi_collective_operations.c', 'w') as f:
        f.write(mpi_code)

    # Step 3: Compile the MPI program using mpicc
    !mpicc -o mpi_collective_operations mpi_collective_operations.c

    # Step 4: Run the compiled MPI program with 4 processes and oversubscription
    !mpirun --oversubscribe -np 4 ./mpi_collective_operations
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Modify the MPI Collective Operations Example

        In this exercise, you will modify the existing MPI program to better understand how different collective communication operations work in MPI.

        ### Task: Modify the Vector Size and Experiment with Collective Operations
        The current code works with a vector of size 8, which is evenly distributed across 4 processes. Modify the program so that:
        1. The vector size is **not evenly divisible** by the number of processes (e.g., change the vector size to 10).
        2. Update the **MPI_Scatter** and **MPI_Gather** operations to handle this uneven distribution properly. This will require adjusting how chunks of the vector are scattered and gathered.

        ### Hints:
        - You can use **`MPI_Scatterv`** and **`MPI_Gatherv`** to handle uneven distributions by specifying the size of each chunk explicitly.
        - The **root process** will still initialize the full vector, and the gathered result should be displayed correctly at the end.

        After making these changes, run the program and observe how the collective operations work with a vector size that isn't divisible by the number of processes.

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

