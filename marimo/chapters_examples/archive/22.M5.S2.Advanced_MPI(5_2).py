import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Dynamic Process Management in MPI
        In MPI, dynamic process management allows an MPI program to spawn new processes during execution, which is critical in scenarios where the workload changes dynamically. The key function for this purpose is MPI_Comm_spawn, which is used to create new processes while the program is running.

        In this example, the parent process (rank 0) will spawn two child processes using the MPI_Comm_spawn function. These child processes perform some work, and the parent process sends a message to the first child via an intercommunicator, which allows communication between the parent and child processes.

        ##Key Functions
        - MPI_Comm_spawn: Spawns new processes dynamically during the execution of the MPI program.
        - MPI_Send: Sends a message from the parent process to the child processes.
        - MPI_Comm_rank: Determines the rank (process ID) within the current communicator.
        - MPI_Comm_size: Determines the total number of processes in the communicator.

        ###In this example, we will:

        - Spawn two child processes from the parent process.
        - Send a message from the parent to one of the child processes.
        - Observe how the child processes receive the message and perform their work.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the parent and child programs to files
    parent_code = \"\"\"
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm intercomm;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0) {
            // Parent process spawns 2 child processes
            MPI_Comm_spawn(\"child_program\", MPI_ARGV_NULL, 2, MPI_INFO_NULL, 0, MPI_COMM_SELF, &intercomm, MPI_ERRCODES_IGNORE);
            printf(\"Parent process spawned child processes.\\n\");

            // Parent sends message to children via intercommunicator
            int msg = 42;
            MPI_Send(&msg, 1, MPI_INT, 0, 0, intercomm); // Send to child 0
        } else {
            // Worker processes perform normal work
            printf(\"Worker process %d doing work.\\n\", rank);
        }

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    child_code = \"\"\"
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0) {
            // Child 0 receives message from the parent
            MPI_Comm parent_comm;
            MPI_Comm_get_parent(&parent_comm);
            int msg;
            MPI_Recv(&msg, 1, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);
            printf(\"Child 0 received message from parent: %d\\n\", msg);
        } else {
            printf(\"Child %d doing work.\\n\", rank);
        }

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Save the parent and child programs to files
    with open(\"parent_program.c\", \"w\") as parent_file:
        parent_file.write(parent_code)

    with open(\"child_program.c\", \"w\") as child_file:
        child_file.write(child_code)

    # Compile the C programs
    !mpicc -o parent_program parent_program.c
    !mpicc -o child_program child_program.c

    # Set environment variables to allow running as root
    import os
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Run the MPI program with 1 parent process, allowing oversubscription
    !mpirun --allow-run-as-root --oversubscribe -np 1 ./parent_program
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Code Walkthrough
        ### Parent Program:
        - MPI_Comm_spawn: The parent process (rank 0) uses MPI_Comm_spawn to dynamically create two child processes. This creates an intercommunicator that allows communication between the parent and child processes.

        - MPI_Send: The parent process sends a message (msg = 42) to the first child (child 0) using the intercommunicator.

        - MPI_Finalize: After sending the message and spawning the children, the parent process finalizes the MPI environment.

        ### Child Program:
        - MPI_Comm_get_parent: Each child process uses this function to get the intercommunicator that connects it to the parent process.

         -MPI_Recv: Child 0 receives the message from the parent. The other child (child 1) simply prints that it is performing some work.

        - MPI_Finalize: Each child process finalizes its MPI environment when the work is done.

        ### Output:
        The output of the program will show the parent process spawning two child processes and sending a message to child 0. Child 0 will print the received message, and child 1 will simply indicate that it is doing work.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Dynamic Process Management: Load Balancing

        In **MPI** (Message Passing Interface), dynamic process management and load balancing are crucial when dealing with varying workloads during runtime. As computational demands change, more processes can be dynamically added to handle the load. This technique is especially useful in simulations or parallel applications where different regions or tasks may require more processing power at different times.

        ### Load Balancing with Dynamic Processes

        Load balancing ensures that tasks are distributed efficiently across processes to maximize resource usage. Some key points about load balancing and dynamic process management in MPI include:

        - **Dynamic task allocation**: Tasks can be dynamically allocated to new processes as needed.
        - **Workload adjustment**: The process resources are adjusted based on the current computational load.
        - **Task redistribution**: Tasks can be redistributed among existing and new processes during execution.
        - **Heterogeneous environments**: This method is ideal for environments where the computational power varies across different hardware (e.g., CPUs, GPUs).
        - **Scaling**: You can dynamically scale the number of processes up or down to handle increasing or decreasing workload efficiently.

        In the following example, the parent process spawns two child processes to handle additional tasks dynamically. The parent assigns different tasks to each of these child processes and sends tasks to the workers through **intercommunication**.

        ### Key Functions:
        - **MPI_Comm_spawn**: Used to dynamically spawn child processes during runtime.
        - **MPI_Send**: Sends data from the parent to the child processes.
        - **MPI_Recv**: Receives data in the child processes from the parent.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing the parent program to a file
    parent_code = \"\"\"
    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm intercomm;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0) {
            // Parent process dynamically spawns 2 more workers
            int extra_workers = 2;
            MPI_Comm_spawn(\"worker_program\", MPI_ARGV_NULL, extra_workers, MPI_INFO_NULL, 0, MPI_COMM_SELF, &intercomm, MPI_ERRCODES_IGNORE);
            printf(\"Parent process spawning extra workers for load balancing.\\n\");

            // Example of task workloads
            int tasks[2] = {10, 20};  // Two different tasks

            // Parent sends different tasks to each child process
            MPI_Send(&tasks[0], 1, MPI_INT, 0, 0, intercomm);  // Task for child 0
            MPI_Send(&tasks[1], 1, MPI_INT, 1, 0, intercomm);  // Task for child 1
        }

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Writing the worker (child) program to a file
    worker_code = \"\"\"
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm parent_comm;
        int task;

        // Get the parent communicator
        MPI_Comm_get_parent(&parent_comm);

        // Each worker process receives a task from the parent
        MPI_Recv(&task, 1, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);
        printf(\"Worker process %d doing task %d\\n\", rank, task);

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Save the parent and worker programs to files
    with open(\"parent_program.c\", \"w\") as parent_file:
        parent_file.write(parent_code)

    with open(\"worker_program.c\", \"w\") as worker_file:
        worker_file.write(worker_code)

    # Compile the parent and worker programs
    !mpicc -o parent_program parent_program.c
    !mpicc -o worker_program worker_program.c

    # Set environment variables to allow running as root
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Run the MPI program with 1 parent process, allowing dynamic spawning of 2 children
    !mpirun --allow-run-as-root --oversubscribe -np 1 ./parent_program
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Code

        In this example, we demonstrate **dynamic process management** and **load balancing** using MPI.

        1. **Parent Process**:
           - The parent process begins by initializing the MPI environment and spawning two child processes using `MPI_Comm_spawn`. The function creates an intercommunicator, which allows communication between the parent and the child processes.
           - The parent assigns two different tasks (represented by the values `10` and `20`) to each child process by sending these tasks through `MPI_Send` via the intercommunicator. Each child receives its respective task and processes it.

        2. **Worker (Child) Processes**:
           - Each worker process (child) receives a task from the parent using `MPI_Recv`. The `MPI_Comm_get_parent` function is used by the child processes to get the intercommunicator that connects them to the parent.
           - The worker processes print out their assigned task and proceed with their work.

        ### Output:
        The output will show the parent process spawning two child processes and assigning each one a different task:

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # One-Sided Communication in MPI

        In **MPI**, one-sided communication allows one process to directly access the memory of another process without requiring explicit cooperation from the target process. This is done using **Remote Memory Access (RMA)** operations, which include functions like `MPI_Put`, `MPI_Get`, and `MPI_Accumulate`.

        ### One-Sided Communication Overview

        One-sided communication is useful in applications where processes need to frequently update or access shared data. Unlike traditional two-sided communication (e.g., `MPI_Send` and `MPI_Recv`), where both the sender and receiver need to actively participate in the communication, one-sided communication allows a process to write to or read from another process's memory independently.

        ### Key Functions
        - **MPI_Win_create**: Defines a window of memory that can be accessed by other processes.
        - **MPI_Put**: A process writes data into the memory of another process.
        - **MPI_Win_fence**: Synchronizes RMA operations to ensure data consistency.
        - **MPI_Win_free**: Frees the memory window when the communication is complete.

        In this example, Process 0 writes a value to the memory of Process 1 using **MPI_Put**. Process 1 exposes its memory using a window, allowing Process 0 to write directly to it.


        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing the updated MPI one-sided communication example to a file
    mpi_code_updated = \"\"\"
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Initialize window and buffer for one-sided communication
        int data;
        MPI_Win win;

        if (rank == 0) {
            // Process 0 writes data to Process 1's memory
            int value_to_put = 42;

            // No memory to expose in Process 0, just creating a window for synchronization
            MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

            // Synchronize before starting the RMA operation
            MPI_Win_fence(0, win);

            // Write value to Process 1's memory at offset 0
            MPI_Put(&value_to_put, 1, MPI_INT, 1, 0, 1, MPI_INT, win);

            // Synchronize after the RMA operation
            MPI_Win_fence(0, win);
        } else if (rank == 1) {
            // Process 1 exposes its memory for Process 0 to write into
            int target_data = 0;
            MPI_Win_create(&target_data, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

            // Synchronize before the RMA operation
            MPI_Win_fence(0, win);

            // Wait for Process 0 to complete the put operation
            MPI_Win_fence(0, win);

            // Process 1 retrieves the data written by Process 0
            printf(\"Process 1 received data: %d\\n\", target_data);
        }

        MPI_Win_free(&win);
        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Save the updated C program to a file
    with open(\"mpi_one_sided_sync.c\", \"w\") as mpi_file:
        mpi_file.write(mpi_code_updated)

    # Compile the C program
    !mpicc -o mpi_one_sided_sync mpi_one_sided_sync.c

    # Set environment variables to allow running as root
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Run the MPI program with 2 processes, allowing one-sided communication
    !mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_one_sided_sync
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Code

        This example demonstrates **one-sided communication** between two processes using MPI.

        1. **Process 0**:
           - Process 0 writes a value (`42`) into the memory of Process 1 using **MPI_Put**.
           - It creates a window using `MPI_Win_create` with `MPI_BOTTOM` as the memory location, meaning that Process 0 does not expose any memory of its own, but rather writes into Process 1's memory.
           - The function **MPI_Put** allows Process 0 to place the value into the memory of Process 1.

        2. **Process 1**:
           - Process 1 exposes its memory for writing by creating a window (`MPI_Win_create`). The `target_data` variable holds the memory that will receive the value from Process 0.
           - After the `MPI_Win_fence` call, Process 1 checks the value that has been written into its memory.

        3. **MPI_Win_fence**:
           - The `MPI_Win_fence` call synchronizes the memory operations. Both Process 0 and Process 1 must call this function to ensure that the memory update by Process 0 is completed before Process 1 attempts to access it.

        ### Output:

        The expected output will show Process 1 receiving the value written by Process 0:

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Synchronization in One-Sided Communication in MPI

        In **MPI**, one-sided communication allows processes to directly read from or write to the memory of another process. Synchronization is crucial when performing **Remote Memory Access (RMA)** operations to ensure data consistency and prevent race conditions.

        ### Synchronization Methods in One-Sided Communication

        - **MPI_Win_fence**: This is a simple synchronization method that acts as a barrier for RMA operations. It ensures that all communication completes before the next operation begins. Each process participating in RMA calls `MPI_Win_fence` to mark the start and end of the communication epoch.
  
        - **MPI_Win_lock and MPI_Win_unlock**: These functions are used to lock memory for exclusive or shared access, preventing race conditions during RMA operations. However, this example will focus on `MPI_Win_fence`.

        In the following example, Process 0 will write a value to Process 1's memory using **MPI_Put**, and Process 1 will read the value after synchronization.

        ### Key Functions:
        - **MPI_Win_create**: Defines a window of memory that other processes can access.
        - **MPI_Put**: Allows one process to write data directly to the memory of another process.
        - **MPI_Win_fence**: Synchronizes RMA operations to ensure that the data is available before any process attempts to access it.


        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 1: Write the corrected MPI C program
    mpi_code_corrected = \"\"\"
    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char *argv[]) {
        int rc;
        rc = MPI_Init(&argc, &argv);
        if (rc != MPI_SUCCESS) {
            printf(\"Error initializing MPI.\\n\");
            MPI_Abort(MPI_COMM_WORLD, rc);
        }

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size < 2) {
            if (rank == 0) {
                printf(\"This program requires at least two MPI processes.\\n\");
            }
            MPI_Finalize();
            return 0;
        }

        int *window_data;
        MPI_Win win;

        // Allocate memory for the window and create it
        rc = MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &window_data, &win);
        if (rc != MPI_SUCCESS) {
            printf(\"Error allocating MPI Window.\\n\");
            MPI_Abort(MPI_COMM_WORLD, rc);
        }

        // Initialize window data
        *window_data = 0;

        printf(\"Process %d: MPI Window allocated and initialized to %d.\\n\", rank, *window_data);

        // Start of RMA epoch
        rc = MPI_Win_fence(0, win);
        if (rc != MPI_SUCCESS) {
            printf(\"Error in MPI_Win_fence (start).\\n\");
            MPI_Abort(MPI_COMM_WORLD, rc);
        }

        if (rank == 0) {
            int value = 42;
            printf(\"Process %d: Putting value %d to process %d's window.\\n\", rank, value, 1);
            rc = MPI_Put(&value, 1, MPI_INT, 1, 0, 1, MPI_INT, win);
            if (rc != MPI_SUCCESS) {
                printf(\"Error in MPI_Put.\\n\");
                MPI_Abort(MPI_COMM_WORLD, rc);
            }
        }

        // End of RMA epoch
        rc = MPI_Win_fence(0, win);
        if (rc != MPI_SUCCESS) {
            printf(\"Error in MPI_Win_fence (end).\\n\");
            MPI_Abort(MPI_COMM_WORLD, rc);
        }

        if (rank == 1) {
            printf(\"Process %d received data: %d\\n\", rank, *window_data);
        }

        // Cleanup
        MPI_Win_free(&win);
        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Save the MPI C program to a file
    with open(\"mpi_one_sided_corrected.c\", \"w\") as mpi_file:
        mpi_file.write(mpi_code_corrected)

    print(\"MPI C program written to 'mpi_one_sided_corrected.c'.\")

    # Step 2: Compile the MPI program
    !mpicc -o mpi_one_sided_corrected mpi_one_sided_corrected.c

    # Check if the executable was created
    if os.path.exists(\"mpi_one_sided_corrected\"):
        print(\"Compilation successful. Executable 'mpi_one_sided_corrected' created.\")
    else:
        print(\"Compilation failed. Please check the C code for errors.\")

    # Step 3: Set environment variables to allow running as root
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    print(\"Environment variables set to allow MPI to run as root.\")

    # Step 4: Execute the MPI program
    print(\"Executing the MPI program...\n\")

    !mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_one_sided_corrected
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Code

        This example demonstrates **one-sided communication** between two processes using **MPI_Win_fence** for synchronization.

        1. **Process 0 (Writer)**:
           - Process 0 creates an RMA window, but it does not use the memory in the window itself; it writes data directly into Process 1's memory.
           - **MPI_Put** is used to write the value `42` into the memory of Process 1.
           - **MPI_Win_fence** is called twice:
             - First to start the RMA epoch before writing data.
             - Second to end the RMA epoch after writing data to ensure synchronization.

        2. **Process 1 (Reader)**:
           - Process 1 creates a window to expose its memory to Process 0.
           - After calling **MPI_Win_fence** (which acts as a synchronization barrier), Process 1 checks the value written by Process 0.
           - The second **MPI_Win_fence** call ensures that Process 1 only reads the value after the data has been fully written by Process 0.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Persistent Communication in MPI

        In **MPI**, persistent communication requests are useful when the communication pattern repeats many times, such as in iterative algorithms (e.g., solving systems of equations with the Jacobi method). Instead of repeatedly setting up and tearing down communication requests, persistent communication allows you to initialize the requests once and reuse them throughout multiple iterations.

        ### Workflow for Persistent Communication

        1. **Initialize**: Create persistent communication requests using `MPI_Send_init` and `MPI_Recv_init`.
        2. **Start**: Begin communication in each iteration using `MPI_Start` or `MPI_Startall` (for multiple requests).
        3. **Wait**: Wait for the communication to complete using `MPI_Wait` or `MPI_Waitall` (for multiple requests).
        4. **Free**: After all iterations, release the resources by calling `MPI_Request_free`.

        This method is especially useful in scenarios where the communication pattern is the same across many iterations. It minimizes the overhead of repeatedly setting up and tearing down requests.

        ### Example Use Case:
        A typical use case for persistent communication is in iterative solvers, like the **Jacobi method**, where each process exchanges boundary data with its neighbors in each iteration.


        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing the MPI persistent communication example to a file
    mpi_code_persistent = \"\"\"
    #include <mpi.h>
    #include <stdio.h>

    void compute_step() {
        // Simulated computation step (could be anything like a Jacobi iteration)
        printf(\"Performing computation step...\\n\");
    }

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int max_iters = 5;
        int neighbor = (rank + 1) % 2;  // Simple neighbor setup for 2 processes
        int send_data = rank + 1;       // Data to send (each process sends its rank+1)
        int recv_data = 0;              // Buffer to receive data
        MPI_Request send_req, recv_req;

        // Initialize persistent communication requests
        MPI_Send_init(&send_data, 1, MPI_INT, neighbor, 0, MPI_COMM_WORLD, &send_req);
        MPI_Recv_init(&recv_data, 1, MPI_INT, neighbor, 0, MPI_COMM_WORLD, &recv_req);

        // Enter the iterative computation loop
        for (int iter = 0; iter < max_iters; iter++) {
            // Start communication requests
            MPI_Startall(2, (MPI_Request[]){send_req, recv_req});

            // Wait for communication to complete
            MPI_Waitall(2, (MPI_Request[]){send_req, recv_req}, MPI_STATUS_IGNORE);

            // Perform computation (can be any function)
            compute_step();

            // Display received data
            printf(\"Rank %d received data: %d in iteration %d\\n\", rank, recv_data, iter);
        }

        // Free persistent communication requests
        MPI_Request_free(&send_req);
        MPI_Request_free(&recv_req);

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Save the C program to a file
    with open(\"mpi_persistent.c\", \"w\") as mpi_file:
        mpi_file.write(mpi_code_persistent)

    # Compile the C program
    !mpicc -o mpi_persistent mpi_persistent.c

    # Set environment variables to allow running as root
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Run the MPI program with 2 processes, allowing persistent communication
    !mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_persistent
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Code

        This example demonstrates **persistent communication** between two processes. The processes send and receive data between themselves in multiple iterations, using **MPI_Send_init** and **MPI_Recv_init** to set up persistent communication requests.

        1. **Initialization**:
           - The program initializes persistent send and receive requests with `MPI_Send_init` and `MPI_Recv_init`.
           - Each process will send its `rank + 1` to its neighbor (i.e., Process 0 sends 1, Process 1 sends 2).

        2. **Communication in Iterations**:
           - In each iteration, `MPI_Startall` is used to start both the send and receive operations.
           - `MPI_Waitall` ensures that the communication is complete before the next computation step.
           - The `compute_step()` function is a placeholder for any actual computation that would be done after each communication round.

        3. **Freeing Requests**:
           - After completing all iterations, the program frees the persistent communication requests using `MPI_Request_free`.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Non-Blocking Point-to-Point Communication in MPI

        In **MPI**, non-blocking communication allows processes to continue computation while the communication is being performed in the background. This is in contrast to blocking communication, where a process waits until the communication is complete before proceeding.

        ### Functions:
        - **MPI_Isend**: Initiates a non-blocking send operation.
        - **MPI_Irecv**: Initiates a non-blocking receive operation.
        - **MPI_Wait**: Ensures that the non-blocking operation completes.

        ### Advantages of Non-Blocking Communication:
        - Non-blocking communication allows processes to overlap computation with communication, improving performance by reducing idle time.
        - It is particularly useful in situations where communication may take a significant amount of time, such as in large distributed systems.

        ### Example Overview:
        - **Process 0** sends its rank to **Process 1** using `MPI_Isend`.
        - Both processes perform some simulated computation while the communication is happening.
        - After completing the computation, they use `MPI_Wait` to ensure the communication is complete before proceeding.

        In this example, **Process 1** will receive data from **Process 0**, while both perform computation in parallel to the data transfer.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing the updated MPI non-blocking point-to-point communication example to a file
    mpi_code_nonblocking = \"\"\"
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int buffer_send = rank + 1;  // Modify the sent value to be rank + 1
        int buffer_recv = 0;
        MPI_Request req_send, req_recv;

        if (rank == 0) {
            // Non-blocking send from Process 0 to Process 1
            MPI_Isend(&buffer_send, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &req_send);
            // Simulated computation after initiating the send
            for (int i = 0; i < 100000; i++) {}  // Simulated computation
            MPI_Wait(&req_send, MPI_STATUS_IGNORE);  // Wait for send to complete
            printf(\"Process 0 finished sending.\\n\");
        } else if (rank == 1) {
            // Non-blocking receive by Process 1 from Process 0
            MPI_Irecv(&buffer_recv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_recv);
            // Simulated computation after initiating the receive
            for (int i = 0; i < 100000; i++) {}  // Simulated computation
            MPI_Wait(&req_recv, MPI_STATUS_IGNORE);  // Wait for receive to complete
            printf(\"Received: %d from Process 0\\n\", buffer_recv);  // Print the received value
        }
        MPI_Finalize();
    }
    \"\"\"

    # Save the C program to a file
    with open(\"mpi_nonblocking.c\", \"w\") as mpi_file:
        mpi_file.write(mpi_code_nonblocking)

    # Compile the C program
    !mpicc -o mpi_nonblocking mpi_nonblocking.c

    # Set environment variables to allow running as root
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Run the MPI program with 2 processes, demonstrating non-blocking communication
    !mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_nonblocking
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Code

        This example demonstrates **non-blocking point-to-point communication** between two processes using `MPI_Isend` and `MPI_Irecv`.

        1. **Process 0 (Sender)**:
           - **Non-blocking Send**: `MPI_Isend` is used to initiate a non-blocking send of its rank to **Process 1**.
           - **Simulated Computation**: While the send is happening, **Process 0** performs some computation (simulated by a loop).
           - **MPI_Wait**: After the computation, **Process 0** waits for the send operation to complete using `MPI_Wait`.

        2. **Process 1 (Receiver)**:
           - **Non-blocking Receive**: `MPI_Irecv` is used to initiate a non-blocking receive of the data from **Process 0**.
           - **Simulated Computation**: While the receive is happening, **Process 1** also performs some computation.
           - **MPI_Wait**: After the computation, **Process 1** waits for the receive operation to complete using `MPI_Wait`.



        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Non-Blocking Collective Operations in MPI

        In **MPI**, non-blocking collective operations allow processes to start a collective operation, such as a broadcast or reduction, and continue performing computation while waiting for the operation to complete. This helps avoid the bottlenecks that can arise when processes wait for each other to complete collective operations.

        ### Benefits of Non-Blocking Collectives:
        - Processes do not have to wait for the collective operation to complete before proceeding with computation.
        - It is particularly useful in applications where collective operations involve large datasets or where synchronization between processes can cause delays.

        ### Functions:
        - **MPI_Ibcast**: Non-blocking version of `MPI_Bcast`, used to broadcast data from one process to all others.
        - **MPI_Ireduce**: Non-blocking version of `MPI_Reduce`, used to reduce data from all processes to a single result (e.g., sum, max).
        - **MPI_Wait**: Used to ensure the collective operation has completed before using the results.

        ### Example Overview:
        In this example, we will use **MPI_Ireduce** to perform a non-blocking reduction operation. While the reduction is happening, the processes will perform some simulated computation. After the computation is done, the program will use `MPI_Wait` to ensure the reduction is complete before accessing the result.

        The reduction operation sums the values from all processes, and the result will be stored in the root process (Process 0).

        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing the MPI non-blocking collective operation example to a file
    mpi_code_nonblocking_collective = \"\"\"
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int sendbuf = rank + 1;  // Each process sends its rank + 1
        int recvbuf = 0;         // The result will be stored in the root (rank 0)
        MPI_Request req;

        // Non-blocking reduction (sum) operation
        MPI_Ireduce(&sendbuf, &recvbuf, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &req);

        // Simulated computation while the reduction is happening
        for (int i = 0; i < 100000; i++) {
            // Perform some dummy computation
            if (i % 20000 == 0 && rank == 0) {
                printf(\"Process 0 doing computation during reduction...\\n\");
            }
        }

        // Wait for the non-blocking reduction to complete
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        // Root process prints the result
        if (rank == 0) {
            printf(\"The sum of ranks is: %d\\n\", recvbuf);
        }

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Save the C program to a file
    with open(\"mpi_nonblocking_collective.c\", \"w\") as mpi_file:
        mpi_file.write(mpi_code_nonblocking_collective)

    # Compile the C program
    !mpicc -o mpi_nonblocking_collective mpi_nonblocking_collective.c

    # Set environment variables to allow running as root
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Run the MPI program with 4 processes, demonstrating non-blocking reduction
    !mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_nonblocking_collective
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Code

        This example demonstrates how to use **non-blocking collective operations** in MPI, specifically `MPI_Ireduce`, to perform a reduction operation while continuing computation.

        1. **Reduction Operation**:
           - Each process sends its rank + 1 to the reduction operation.
           - `MPI_Ireduce` is used to sum the values across all processes, and the result is stored in **Process 0**.
           - Since this is a non-blocking operation, the processes do not wait for the reduction to complete immediately.

        2. **Simulated Computation**:
           - While the reduction is happening in the background, the processes perform some dummy computation. In this case, the computation is simulated with a simple loop.
           - **Process 0** prints messages during the computation to show that it is performing work while the reduction is ongoing.

        3. **Waiting for Completion**:
           - After the computation is done, `MPI_Wait` is called to ensure that the non-blocking reduction has completed.
           - Once the reduction is complete, **Process 0** prints the result of the reduction.

        ### Output:
        The expected output will show that **Process 0** performs computation while the reduction is happening:

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Message Aggregation in MPI

        **Message aggregation** is a technique used to reduce communication overhead by combining multiple small messages into a larger message. This minimizes the number of times the startup latency of communication is incurred, especially in high-latency networks. Instead of sending multiple small messages, we combine them into a single structure and send it as one message.

        ### Benefits of Message Aggregation:
        - Reduces the number of communication calls.
        - Reduces the startup latency associated with each message.
        - Optimizes the use of bandwidth by sending larger, aggregated messages.

        ### Example:
        In this example, we will define a `data_packet` structure that contains information about temperature, pressure, and humidity. Instead of sending these values individually, we will aggregate them into a single packet and send them as a **binary message** (`MPI_BYTE`) from **Process 0** to **Process 1**.

        The structure we will use is:
        ```c
        struct data_packet {
            double temperature;
            double pressure;
            double humidity;
        };

        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing the MPI message aggregation example to a file
    mpi_code_message_aggregation = \"\"\"
    #include <mpi.h>
    #include <stdio.h>

    // Define the data_packet structure
    struct data_packet {
        double temperature;
        double pressure;
        double humidity;
    };

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        struct data_packet packet;

        if (rank == 0) {
            // Process 0 prepares the data to send
            packet.temperature = 23.4;
            packet.pressure = 1013.5;
            packet.humidity = 45.6;

            // Send the aggregated message as a single binary message (MPI_BYTE)
            MPI_Send(&packet, sizeof(packet), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
            printf(\"Process 0 sent data: Temp=%.1f, Pressure=%.1f, Humidity=%.1f\\n\",
                   packet.temperature, packet.pressure, packet.humidity);
        } else if (rank == 1) {
            // Process 1 receives the data_packet structure
            MPI_Recv(&packet, sizeof(packet), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf(\"Process 1 received data: Temp=%.1f, Pressure=%.1f, Humidity=%.1f\\n\",
                   packet.temperature, packet.pressure, packet.humidity);
        }

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Save the C program to a file
    with open(\"mpi_message_aggregation.c\", \"w\") as mpi_file:
        mpi_file.write(mpi_code_message_aggregation)

    # Compile the C program
    !mpicc -o mpi_message_aggregation mpi_message_aggregation.c

    # Set environment variables to allow running as root
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Run the MPI program with 2 processes, demonstrating message aggregation
    !mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_message_aggregation
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Code

        This example demonstrates **message aggregation** by sending a single structure containing multiple data fields from **Process 0** to **Process 1**. The structure contains three fields: temperature, pressure, and humidity.

        1. **Structure Definition**:
           - We define a structure `data_packet` that contains three `double` values: `temperature`, `pressure`, and `humidity`.

        2. **Process 0 (Sender)**:
           - **Process 0** initializes the structure with specific values for temperature, pressure, and humidity.
           - The structure is sent to **Process 1** using `MPI_Send`. Instead of sending each field individually, the entire structure is sent as a single message using the `MPI_BYTE` data type, which treats the structure as a raw block of memory.

        3. **Process 1 (Receiver)**:
           - **Process 1** receives the entire structure in one go using `MPI_Recv`. The received structure is unpacked directly into a `data_packet` variable.
           - **Process 1** prints the received temperature, pressure, and humidity values.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Topology-Aware Communication in MPI

        **Topology-aware communication** helps improve the performance of parallel applications by mapping MPI processes to match the underlying hardware topology. This reduces the number of communication "hops" between processes, which is particularly important in large-scale systems where network latency can affect performance.

        ### Benefits of Topology-Aware Communication:
        - Aligns the logical MPI process grid with the physical hardware layout.
        - Minimizes communication distance (number of hops) between processes.
        - Reduces network congestion and improves overall application performance.

        ### Function:
        - **MPI_Cart_create**: Creates a Cartesian grid topology that organizes processes into a structured grid. This allows for efficient neighbor communication, such as in simulations that involve grids or meshes (e.g., computational fluid dynamics).

        ### Example Overview:
        In this example, we will create a 2D Cartesian grid of processes using `MPI_Cart_create`. The grid will help map processes logically and reduce the number of hops between communicating neighbors.

        - **dims**: Specifies the dimensions of the Cartesian grid (e.g., 2D grid with `x_size` and `y_size`).
        - **periods**: Specifies whether the grid should have periodic boundaries (e.g., for toroidal grids).
        - **cart_comm**: The communicator that will represent the new Cartesian grid.

        We will then print the coordinates of each process in the Cartesian grid.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing the MPI topology-aware communication example to a file
    mpi_code_topology_aware = \"\"\"
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Define dimensions of the Cartesian grid (2D grid for simplicity)
        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);  // Automatically compute grid dimensions

        // No periodic boundaries
        int periods[2] = {0, 0};
        MPI_Comm cart_comm;

        // Create the Cartesian grid topology
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

        // Get the coordinates of each process in the Cartesian grid
        int coords[2];
        MPI_Cart_coords(cart_comm, rank, 2, coords);

        // Print the rank and the Cartesian coordinates of each process
        printf(\"Process %d is at coordinates (%d, %d) in the Cartesian grid.\\n\",
               rank, coords[0], coords[1]);

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Save the C program to a file
    with open(\"mpi_topology_aware.c\", \"w\") as mpi_file:
        mpi_file.write(mpi_code_topology_aware)

    # Compile the C program
    !mpicc -o mpi_topology_aware mpi_topology_aware.c

    # Set environment variables to allow running as root
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Run the MPI program with 4 processes, demonstrating Cartesian grid topology
    !mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_topology_aware
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Code

        This example demonstrates how to use **MPI_Cart_create** to organize MPI processes into a Cartesian grid topology. Each process is placed in a 2D grid, and we print the coordinates of each process within the grid.

        1. **Defining the Grid Dimensions**:
           - `MPI_Dims_create` automatically computes the dimensions of the Cartesian grid based on the total number of processes (`size`). In this case, we are creating a 2D grid.

        2. **Creating the Cartesian Communicator**:
           - `MPI_Cart_create` creates the Cartesian grid topology. The `periods` array specifies whether the grid has periodic boundaries (for example, wrapping around the edges like a toroidal grid). In this case, the grid does not have periodic boundaries.

        3. **Retrieving Process Coordinates**:
           - `MPI_Cart_coords` retrieves the coordinates of each process in the Cartesian grid. This allows us to determine where each process is located within the grid.

        4. **Printing Coordinates**:
           - Each process prints its rank and the corresponding coordinates in the grid.

        ### Output:
        The expected output will show the coordinates of each process within the Cartesian grid:

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to Matrix Multiplication using MPI

        Matrix multiplication is a computationally intensive task, especially when the size of matrices increases. Serial implementations become impractical when dealing with large matrices, making parallelization necessary for efficient computation.

        ### Parallelizing Matrix Multiplication using MPI

        In this section, we will explore how to implement matrix multiplication using Message Passing Interface (MPI). We will begin with a basic parallel implementation where the matrix data is divided and distributed across multiple processes. Then, we will enhance this basic version by introducing advanced MPI features such as:
        - **Dynamic Process Management**: Dynamically adjusting the number of worker processes during execution.
        - **One-Sided Communication**: Using Remote Memory Access (RMA) for asynchronous communication.
        - **Persistent Communication Requests**: Reusing communication handles to reduce overhead in repetitive operations.
        - **Non-Blocking Collective Operations**: Overlapping communication with computation to optimize performance.

        Let's start by implementing the basic parallel matrix multiplication using MPI.

        ### Serial Matrix Multiplication Overview

        In serial matrix multiplication, we compute each element of the result matrix `C` by taking the dot product of a row from matrix `A` and a column from matrix `B`. The following code illustrates this process:

        ```c
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++) {
                C[i][j] = 0;
                for (int k = 0; k < M; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        ```


        However, as the matrix size grows, the memory and computational requirements grow cubically, necessitating parallel computation.

        ### Step 1: Basic MPI for Distributed Matrix Multiplication
        The first step in parallelizing the matrix multiplication is distributing the data across multiple processes. Each process computes a subset of the result matrix C. We use the following MPI functions:

        - MPI_Scatter: Distributes blocks of matrix A to each process.
        - MPI_Bcast: Broadcasts matrix B to all processes.
        - MPI_Gather: Collects the computed blocks of C back to the master process.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing the corrected MPI matrix multiplication example to a file
    mpi_code_matrix_multiplication = r\"\"\"
    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>

    #define MASTER 0

    void initialize_matrix(double* matrix, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i * cols + j] = rand() % 10;  // Initialize with random values
            }
        }
    }

    int main(int argc, char *argv[]) {
        int rank, size, N = 4, M = 4, P = 4, rows_per_proc;
        double *A, *B, *C, *local_A, *C_part;

        MPI_Init(&argc, &argv);                        // Initialize MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);          // Get process rank
        MPI_Comm_size(MPI_COMM_WORLD, &size);          // Get total number of processes

        if (N % size != 0) {
            if (rank == MASTER) {
                printf(\"N (%d) is not divisible by the number of processes (%d).\n\", N, size);
            }
            MPI_Finalize();
            return -1;
        }

        rows_per_proc = N / size;  // Divide the rows among processes

        // Allocate memory for matrix B on all processes
        B = (double*) malloc(M * P * sizeof(double));
        if (B == NULL) {
            printf(\"Process %d: Unable to allocate memory for matrix B.\n\", rank);
            MPI_Finalize();
            return -1;
        }

        // Allocate memory for matrices on MASTER
        if (rank == MASTER) {
            A = (double*) malloc(N * M * sizeof(double));
            C = (double*) malloc(N * P * sizeof(double));

            if (A == NULL || C == NULL) {
                printf(\"MASTER: Unable to allocate memory for matrices A or C.\n\");
                MPI_Finalize();
                return -1;
            }

            initialize_matrix(A, N, M);  // Initialize matrix A
            initialize_matrix(B, M, P);  // Initialize matrix B
        }

        // Allocate memory for local_A and C_part on all processes
        local_A = (double*) malloc(rows_per_proc * M * sizeof(double));
        C_part = (double*) malloc(rows_per_proc * P * sizeof(double));

        if (local_A == NULL || C_part == NULL) {
            printf(\"Process %d: Unable to allocate memory for local_A or C_part.\n\", rank);
            MPI_Finalize();
            return -1;
        }

        // Distribute matrix A
        MPI_Scatter(A, rows_per_proc * M, MPI_DOUBLE,
                    local_A, rows_per_proc * M, MPI_DOUBLE,
                    MASTER, MPI_COMM_WORLD);

        // Broadcast matrix B to all processes
        MPI_Bcast(B, M * P, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

        // Perform local computation of matrix multiplication
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < P; j++) {
                C_part[i * P + j] = 0;
                for (int k = 0; k < M; k++) {
                    C_part[i * P + j] += local_A[i * M + k] * B[k * P + j];
                }
            }
        }

        // Gather the computed parts of matrix C from all processes
        MPI_Gather(C_part, rows_per_proc * P, MPI_DOUBLE,
                   C, rows_per_proc * P, MPI_DOUBLE,
                   MASTER, MPI_COMM_WORLD);

        // Optionally, MASTER can print the result
        if (rank == MASTER) {
            printf(\"\\nMatrix A:\\n\");
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < M; j++) {
                    printf(\"%lf \", A[i * M + j]);
                }
                printf(\"\\n\");
            }

            printf(\"\\nMatrix B:\\n\");
            for(int i = 0; i < M; i++) {
                for(int j = 0; j < P; j++) {
                    printf(\"%lf \", B[i * P + j]);
                }
                printf(\"\\n\");
            }

            printf(\"\\nMatrix C (Result):\\n\");
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < P; j++) {
                    printf(\"%lf \", C[i * P + j]);
                }
                printf(\"\\n\");
            }
        }

        // Free allocated memory
        if (rank == MASTER) {
            free(A);
            free(C);
        }
        free(B);
        free(local_A);
        free(C_part);

        MPI_Finalize();  // Finalize MPI
        return 0;
    }
    \"\"\"

    # Save the corrected C program to a file
    with open(\"mpi_matrix_multiplication.c\", \"w\") as mpi_file:
        mpi_file.write(mpi_code_matrix_multiplication)

    # Compile the C program
    !mpicc -o mpi_matrix_multiplication mpi_matrix_multiplication.c

    # Set environment variables to allow running as root
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Run the MPI program with 4 processes
    !mpirun --allow-run-as-root --oversubscribe -np 4 ./mpi_matrix_multiplication
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of the Basic MPI Matrix Multiplication Code

        1. **MPI Initialization**:
           The program starts with `MPI_Init`, which initializes the MPI environment. Each process retrieves its rank (process identifier) and the total number of processes using `MPI_Comm_rank` and `MPI_Comm_size`.

        2. **Matrix Initialization**:
           Matrix `A` and `B` are initialized by the master process (`rank == MASTER`). These matrices are then distributed to all the worker processes using `MPI_Scatter` and `MPI_Bcast`.

        3. **Matrix Multiplication**:
           Each process computes a part of the resulting matrix `C`. Each process is responsible for multiplying a subset of rows of `A` with the entire matrix `B`.

        4. **Collecting Results**:
           After each process completes its computation, the results are gathered back into the full matrix `C` on the master process using `MPI_Gather`.

        5. **Finalization**:
           The program finalizes the MPI environment using `MPI_Finalize`, and all dynamically allocated memory is freed.

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


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

