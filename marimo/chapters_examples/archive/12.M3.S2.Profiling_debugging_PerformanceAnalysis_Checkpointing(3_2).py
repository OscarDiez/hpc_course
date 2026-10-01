import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Tools for Profiling HPC Applications

        In high-performance computing (HPC), optimizing applications requires a deep understanding of how code interacts with the underlying hardware. Profiling tools provide insights into CPU usage, memory consumption, and performance characteristics. In this section, we explore some widely used tools for CPU and memory profiling, including `gprof`, `perf`, and Intel's VTune.

        ### Gprof and Perf

        **Gprof** is a GNU profiling tool that collects and arranges statistics on program execution. It is useful for identifying functions consuming the most time, making it ideal for CPU-bound applications. However, Gprof might miss short-lived functions or provide limited insights into multi-threaded programs.

        **Perf**, built into the Linux kernel, offers a more comprehensive analysis by providing detailed information about CPU usage, cache hits and misses, and more. Perf supports system-wide and application-specific profiling, making it valuable for identifying bottlenecks in complex HPC environments.

        Example of using `perf` for profiling:
        ```bash
        perf record -F 99 -a -g -- ./my_hpc_application
        perf report

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Tools for Profiling HPC Applications

        In high-performance computing (HPC), it is crucial to understand where computational resources are spent in order to optimize performance. Profiling tools, such as `gprof`, help identify functions that consume the most execution time.

        In this exercise, we will:
        1. Write three functions that perform different matrix operations (multiplication, addition, and transpose).
        2. Use `gprof` to profile the execution and compare the time spent in each function.
        3. Learn how to analyze profiling reports and identify which function is the most computationally expensive.

        By understanding profiling reports, we can focus our optimization efforts on the most time-consuming parts of the code.

        """
    )
    return


app._unparsable_cell(
    r"""
    import os
    import subprocess

    def is_colab():
        # Check if running in Google Colab
        return os.path.exists('/content')

    def is_hpc_cluster():
        # Check if running on an HPC cluster
        return os.path.exists('/cvmfs/soft.computecanada.ca')

    def compile_and_run_gprof_code():
        # Write the C code for profiling with gprof
        code = \"\"\"
        #include <stdio.h>
        #include <stdlib.h>

        #define N 500  // Define matrix size

        // Matrix Multiplication
        void matrix_multiply(int A[N][N], int B[N][N], int C[N][N]) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    C[i][j] = 0;
                    for (int k = 0; k < N; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        // Matrix Addition
        void matrix_add(int A[N][N], int B[N][N], int C[N][N]) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    C[i][j] = A[i][j] + B[i][j];
                }
            }
        }

        // Matrix Transpose
        void matrix_transpose(int A[N][N], int T[N][N]) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    T[j][i] = A[i][j];
                }
            }
        }

        int main() {
            int A[N][N], B[N][N], C[N][N], T[N][N];

            // Initialize matrices A and B with random values
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A[i][j] = rand() % 100;
                    B[i][j] = rand() % 100;
                }
            }

            // Perform matrix operations
            matrix_multiply(A, B, C);      // Matrix Multiplication
            matrix_add(A, B, C);           // Matrix Addition
            matrix_transpose(A, T);        // Matrix Transpose

            return 0;
        }
        \"\"\"

        # Save the C code to a file
        with open(\"matrix_operations.c\", \"w\") as file:
            file.write(code)

        # Compile the C program with profiling enabled (-pg flag)
        if is_colab():
            # In Colab
            !gcc -pg -o matrix_operations matrix_operations.c
            !./matrix_operations
            !gprof matrix_operations gmon.out > profile_report.txt
            !cat profile_report.txt
        elif is_hpc_cluster():
            # In HPC Cluster
            subprocess.run([\"gcc\", \"-pg\", \"-o\", \"matrix_operations\", \"matrix_operations.c\"])
            subprocess.run([\"./matrix_operations\"])
            subprocess.run([\"gprof\", \"./matrix_operations\", \"gmon.out\"], stdout=open('profile_report.txt', 'w'))
            with open('profile_report.txt', 'r') as f:
                print(f.read())

    # Run the function to compile and profile the code
    compile_and_run_gprof_code()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explanation of Profiling Code

        ### What the Code Does

        The C program performs a matrix multiplication of two 500x500 matrices. It is compiled with the `-pg` flag, enabling `gprof` to collect profiling information. This information is saved in a file called `gmon.out`, and `gprof` generates a detailed profiling report that shows how much time was spent in each function.

        The profiling process is as follows:
        1. **Matrix Multiplication**: The program generates random 500x500 matrices, multiplies them, and outputs nothing (focusing only on profiling performance).
        2. **Profiling with `gprof`**: The program is compiled with the `-pg` flag to enable profiling. After running the program, `gprof` is used to generate a profiling report.

        ### Understanding the Report

        The report will show statistics on function calls, including:
        - **Self Time**: The time spent in the function itself.
        - **Total Time**: The total time spent in the function, including calls to other functions.
        - **Call Graph**: A breakdown of which functions were called, and how much time was spent in each.

        ### Learning Points
        - **Profiling in HPC**: Profiling helps identify bottlenecks in HPC applications, especially in compute-heavy functions like matrix multiplication.
        - **Environment-Specific Execution**: This notebook is designed to run in both Google Colab and an HPC cluster, adjusting the commands for each environment.
        - **Next Steps**: Use this information to optimize the matrix multiplication algorithm by implementing techniques like loop unrolling, blocking, or vectorization to improve performance.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Valgrind Memory Profiling Example

        In this section, we will demonstrate how to use **Valgrind**, a powerful dynamic analysis tool, to detect memory-related issues in a C program. Valgrind can identify problems such as memory leaks, invalid memory access, buffer overflows, and other memory mismanagement issues.

        The following example intentionally introduces two common memory issues:
        - **Memory Leak**: We allocate memory for an array but forget to free it, which leads to a memory leak.
        - **Invalid Write**: The program accesses an out-of-bounds memory location (writing past the allocated memory), which Valgrind will catch as an invalid write.

        The Valgrind tool will analyze the program and generate detailed information on these issues, helping us understand what went wrong and where.

        We will:
        1. Compile a simple C program with memory errors.
        2. Run the compiled program through Valgrind's Memcheck tool.
        3. Analyze the Valgrind output to identify the errors.

        Let's begin by running the following code.

        """
    )
    return


app._unparsable_cell(
    r"""
    # If using a cluster with modules (e.g. Magi Castle)
    !bash -c \"module load valgrind-mpi/3.16.1 && module list\"
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # IF no modules try to install it (e/g/ COLAB)
    # Install Valgrind (required for Colab and similar environments)
    !apt-get install -y valgrind
    """,
    name="_"
)


@app.cell
def _(subprocess):
    def run_valgrind_example():
        # C code with memory and logic errors
        code = """
        #include <stdio.h>
        #include <stdlib.h>

        int main() {
            int *array = (int*)malloc(5 * sizeof(int));  // Allocate space for 5 integers

            // Error: Accessing out-of-bounds memory (invalid write)
            for (int i = 0; i <= 5; ++i) {  // The loop should run until i < 5
                array[i] = i * 10;
            }

            // Printing the array
            printf("Array contents:\\n");
            for (int i = 0; i < 5; ++i) {
                printf("%d ", array[i]);
            }
            printf("\\n");

            // Error: Forgetting to free the memory (memory leak)
            // free(array);  <-- This line is commented, leading to a memory leak

            return 0;
        }
        """

        # Save the C code to a file
        with open("valgrind_example.c", "w") as file:
            file.write(code)

        # Compile the C program
        compile_result = subprocess.run(["gcc", "-o", "valgrind_example", "valgrind_example.c"], capture_output=True, text=True)

        if compile_result.returncode != 0:
            print(f"Compilation failed:\n{compile_result.stderr}")
            return
        else:
            print("Compilation successful.")

        # Run the program with Valgrind's Memcheck tool to detect memory errors
        valgrind_command = [
            "valgrind",
            "--leak-check=full",           # Detailed memory leak detection
            "--track-origins=yes",         # Track where uninitialized values come from
            "--show-reachable=yes",        # Show all reachable memory at the end
            "--log-file=valgrind_log.txt", # Save Valgrind output to a file
            "./valgrind_example"
        ]

        valgrind_result = subprocess.run(valgrind_command, capture_output=True, text=True)

        if valgrind_result.returncode != 0:
            print(f"Valgrind execution failed:\n{valgrind_result.stderr}")
        else:
            print("Valgrind execution output:\n")
            with open("valgrind_log.txt", "r") as log_file:
                print(log_file.read())

    # Run the function to compile and execute the C program under Valgrind
    run_valgrind_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Valgrind Output Explanation

        The Valgrind Memcheck tool analyzed the program and detected two critical issues:

        1. **Invalid Write**:
            - Valgrind reports an **invalid write** of size 4 (attempting to write an integer).
            - This happens because the loop accesses memory beyond the allocated array. The array has 5 elements (index 0 to 4), but the loop tries to access index 5 (`i <= 5` should be `i < 5`).
            - Valgrind provides the exact line where the error occurred and the size of the block of memory allocated (`20 bytes for 5 integers`).

        2. **Memory Leak**:
            - Valgrind detects that 20 bytes of memory were **definitely lost**. This means that memory was allocated but never freed, leading to a memory leak.
            - Valgrind's **Leak Summary** shows that the program allocated memory but did not free it (since `free(array);` was commented out).

        This example demonstrates the power of Valgrind in detecting common memory errors that can lead to crashes or inefficient memory usage in programs. Valgrind is particularly useful in ensuring memory safety and identifying bugs that can otherwise be difficult to catch through regular debugging.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Executing Serial and Parallel Profiling on an HPC Cluster with TAU

        Profiling is a critical technique used in High-Performance Computing (HPC) to identify bottlenecks and optimize code performance. This guide will walk you through the steps to execute serial and parallel profiling on an HPC cluster using **TAU** (Tuning and Analysis Utilities).

        We will use **matrix multiplication** as an example for both serial and parallel profiling.

        ---

        ## Serial Profiling: Step-by-Step

        ### 1. Loading the Required Modules

        Before running TAU, you need to load the necessary modules. TAU is typically available on HPC systems as a module. To check if TAU is available and load it, run the following commands:

        ```bash
        module avail tau   # Check if TAU is available on your cluster
        module load tau/2.30.1  # Load the required version of TAU

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ---

        ### **Step 2: Serial Matrix Multiplication Program (C)**

        The following C code performs matrix multiplication for two 500x500 matrices.

        ```c
        #include <stdio.h>

        #define N 500  // Size of the matrix

        void matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    C[i][j] = 0;
                    for (int k = 0; k < N; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        int main() {
            double A[N][N], B[N][N], C[N][N];
    
            // Initialize matrices A and B
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A[i][j] = i + j;
                    B[i][j] = i - j;
                }
            }

            // Perform matrix multiplication
            matrix_multiply(A, B, C);

            return 0;
        }

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ### **Step 3: Compiling the Serial Program**

        Use TAU's compiler to instrument the code for profiling:

        ```bash
        tau_cc -o matrix_serial matrix_multiply.c
        ```

        This will compile the serial matrix multiplication program and prepare it for profiling.

        ---

        ### **Step 4: Running the Serial Program**

        Now, run the compiled program:

        ```bash
        ./matrix_serial
        ```

        This will generate a profiling log of the program’s execution.

        ---

        ### **Step 5: Viewing the Profiling Results**

        Once the program finishes running, use TAU’s `pprof` tool to view the profiling results:


        You will see a table showing the breakdown of where the program spent its time. The output might look like this:

        ```bash
        % cumulative self self total time seconds seconds calls ms/call ms/call name 75.0 0.30 0.30 1 300.00 400.00 matrix_multiply 25.0 0.40 0.10 1 100.00 100.00 initialize_matrix
        ```

        ---

        ### **Step 6: Analysis**

        The output shows that the `matrix_multiply` function takes the majority of the program’s execution time. This function can be optimized by techniques such as:

        - **Loop Unrolling**: Reduces loop overhead and increases instruction-level parallelism.
        - **Cache Blocking**: Improves cache usage by breaking down the matrix into smaller blocks.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Parallel Matrix Multiplication Code
        ### **Step 1: Writing the Parallel Matrix Multiplication Code**

        The following code demonstrates a parallel matrix multiplication implementation using MPI. This code will be used to perform profiling on a parallel workload:

        ```c
        #include <mpi.h>
        #include <stdio.h>
        #include <stdlib.h>

        #define N 4  // Matrix size (N x N)

        void matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    C[i][j] = 0;
                    for (int k = 0; k < N; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        int main(int argc, char** argv) {
            MPI_Init(&argc, &argv);  // Initialize MPI

            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            double A[N][N], B[N][N], C[N][N];

            // Initialize matrices A and B
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A[i][j] = rank + i + j;
                    B[i][j] = rank + i - j;
                }
            }

            // Perform matrix multiplication
            matrix_multiply(A, B, C);

            // Print the result matrix from process 0
            if (rank == 0) {
                printf("Matrix C (result):\n");
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        printf("%f ", C[i][j]);
                    }
                    printf("\n");
                }
            }

            MPI_Finalize();  // Finalize MPI
            return 0;
        }

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### **Step 2: Compiling the Parallel Program**

        To instrument the code for profiling with TAU and compile it, use the following command:

        ```bash
        tau_cc -T mpi -o matrix_parallel mpi_matrix_multiply.c
        ```

        This will compile the program and instrument it for parallel profiling.

        Step 3: Running the Parallel Program with TAU
        Run the instrumented parallel program using mpirun and tau_exec to trace the parallel execution:

        ```bash
        tau_exec mpirun --oversubscribe -np 4 ./matrix_parallel
        ```

        The --oversubscribe flag allows more processes than the available CPU cores in testing environments. After executing the program, TAU will record profiling data for each MPI process.

        Step 4: Viewing the Profiling Data
        Once the program finishes, you can view the profiling data using TAU’s paraprof tool, which provides a GUI interface to visualize the results:

        ```bash
        paraprof
        ```

        In case you encounter issues with Java, ensure that the correct version of Java is loaded using the appropriate module (e.g., module load java/17.0.2).

        Step 5: Analyzing the Profiling Output
        TAU will generate a detailed report that includes information about MPI calls and the time spent in each function. An example of the analysis output might look like this:

        ```bash
        Process 0: Time spent in MPI_Wait: 50%
        Process 1: Time spent in MPI_Wait: 5%
        Process 2: Time spent in MPI_Wait: 5%
        ```

        This suggests that process 0 is spending a significant amount of time waiting, likely due to load imbalance. Improving load balancing by redistributing matrix blocks or optimizing communication can enhance performance.

        Step 6: Optimization Suggestions
        The profiling results indicate that the workload is not evenly distributed across the processes. To address this:

        Redistribute Matrix Blocks: Split matrix blocks evenly across processes to ensure that no single process does more work than others.
        Optimize Communication: Reduce the time spent waiting for data from other processes by optimizing communication between processes using non-blocking MPI calls.
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
        # Performance Profiling in HPC with Gperftools

        ## Introduction

        In this practice, we will explore how to profile applications in a high-performance computing (HPC) environment using **gperftools**. Gperftools is a popular set of performance analysis tools originally developed by Google and widely used in HPC to measure CPU and memory usage in large-scale applications. Profiling is a crucial step in optimizing HPC applications, as it helps developers identify bottlenecks and inefficient code paths, which is especially important in parallel and distributed computing environments.

        We will focus on two main tools in this suite:
        1. **pprof**: A CPU statistical profiler that provides insights into the time spent in different parts of a program.
        2. **tc_malloc**: A thread-caching memory allocator that enhances memory allocation performance and supports dynamic memory profiling.

        By the end of this practice, you will:
        - Learn how to use gperftools to profile a basic C++ application.
        - Understand how to interpret profiling data for performance tuning.
        - Explore profiling in both single-node and multi-node (MPI-based) setups.

        **Why profiling?**
        Profiling helps us to:
        - Identify performance bottlenecks.
        - Optimize the code for better CPU usage.
        - Monitor memory leaks and dynamic memory allocation issues.
        - Enhance scalability by understanding the performance of MPI-based parallel applications.

        ---

        ## Steps for Profiling with Gperftools

        We will use a simple matrix multiplication program as our example to understand how profiling works. This example will be written in C++, and we will:
        1. Compile the program with gperftools profiling enabled.
        2. Run the program to generate a profile report.
        3. Analyze the profiling report using `pprof`.

        **Note:** The profiling tools need to be installed in the environment where the code is run, which we will do step by step in this notebook.

        ---

        ## Installation of Gperftools in Google Colab

        We will first install `gperftools` in Google Colab as it is not natively available.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Locate the directory where pprof is installed
    !which pprof

    # If pprof is not found, try updating the PATH manually (if installed in a non-standard location)
    # Update the path where gperftools is installed
    !export PATH=$PATH:/usr/local/bin:/usr/bin

    # Verify if pprof is now available
    !pprof --version
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Create a C++ source file for matrix multiplication with profiling enabled
    cpp_code = \"\"\"
    #include <iostream>
    #include <vector>
    #include <gperftools/profiler.h>

    using namespace std;

    // Function for matrix multiplication
    void matrixMultiply(int N) {
        vector<vector<int>> A(N, vector<int>(N, 1));
        vector<vector<int>> B(N, vector<int>(N, 1));
        vector<vector<int>> C(N, vector<int>(N, 0));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    int main() {
        ProfilerStart(\"matrixmult.prof\"); // Start profiling
        int N = 1000;  // Matrix size
        matrixMultiply(N);
        ProfilerStop(); // Stop profiling
        return 0;
    }
    \"\"\"

    # Save to a file
    with open(\"matrixmult.cpp\", \"w\") as file:
        file.write(cpp_code)

    # Compile the code with gperftools
    !g++ -O2 -lprofiler matrixmult.cpp -o matrixmult

    # Run the compiled program to generate profiling data
    !env CPUPROFILE=matrixmult.prof ./matrixmult
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Install Graphviz for graphical visualization of profiles
    !apt-get install -y graphviz

    # Run pprof to generate a text-based report
    !pprof --text ./matrixmult matrixmult.prof

    # Optionally, generate a graphical representation of the profile
    !pprof --pdf ./matrixmult matrixmult.prof > matrixmult_profile.pdf
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
        ## Explanation of Code Sections

        ### 1. Installation of Gperftools

        We start by installing `gperftools` in Google Colab. The command `apt-get install google-perftools` installs the necessary libraries for using gperftools, including the CPU profiler `pprof` and memory allocator `tc_malloc`.

        ### 2. Writing and Compiling the Matrix Multiplication Program

        In this example, we created a C++ program that performs matrix multiplication on two square matrices of size `N`. Matrix multiplication is a good candidate for profiling in HPC environments because of its computational intensity and potential memory bottlenecks.

        The program is instrumented with gperftools using the `ProfilerStart` and `ProfilerStop` functions, which enable profiling around the matrix multiplication logic. We use the command `env CPUPROFILE=matrixmult.prof ./matrixmult` to run the program and generate a profile.

        ### 3. Analyzing the Profile

        We then use the `pprof` tool to analyze the generated profiling data. The text-based report produced by `pprof --text` gives us insights into the function execution time. To better visualize the data, we generate a PDF using `pprof --pdf` which contains a graph showing the CPU usage breakdown across different functions.

        The results allow us to see which parts of the matrix multiplication are consuming the most resources, and based on that, we can optimize the code by improving algorithmic efficiency or better utilizing memory.

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Profiling Distributed Applications using Scalasca

        ## Introduction

        In this practice, we will explore **Scalasca**, a performance analysis tool used to profile MPI-based distributed applications. Scalasca helps to identify performance bottlenecks in parallel applications, especially for message-passing systems.

        Profiling is essential in **High-Performance Computing (HPC)** because it allows developers to:
        - Pinpoint inefficiencies in communication or computation.
        - Measure parallel scalability.
        - Optimize code for better CPU and network utilization.

        In this lesson, we will profile a simple MPI-based application and analyze the results using Scalasca.

        ### Why Profiling Matters in HPC

        HPC applications often run on distributed systems, requiring efficient communication between nodes. Profiling tools such as Scalasca allow you to visualize how time is spent in the application, highlighting areas of improvement, including:
        1. **Load imbalance** – Are all processors doing equal work?
        2. **Communication overhead** – How much time is spent waiting for data from other nodes?
        3. **Scalability** – How does the performance change when the number of nodes increases?

        ---

        ## Steps for Profiling a Distributed Application with Scalasca

        1. **Write a simple MPI application**.
        2. **Compile the application with Scalasca support**.
        3. **Run the application on multiple nodes with Scalasca**.
        4. **Analyze the Scalasca performance report**.

        To do this in Google Colab, we will first install an MPI library (`OpenMPI`) and Scalasca, then run the distributed application and simulate multi-node behavior.

        ---

        ## Installing MPI and Scalasca in Google Colab

        Since Google Colab doesn't have MPI and Scalasca installed by default, we'll first install these packages. This setup assumes that we simulate a multi-node environment in Colab, and the actual profiling will be similar to what you would run on a cluster.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of the Code

        ### 1.Loading the module Scalasca

        We first install MPI (OpenMPI) and Scalasca in Google Colab to set up our environment. MPI allows us to run parallel programs over multiple processes, and Scalasca is the tool we use to profile these parallel programs.

        ### 2. Writing the MPI Application

        The MPI application performs a simple **matrix multiplication** distributed across several processes. We use MPI to divide the computation across multiple processors, simulating a real-world distributed computation.

        - **MPI_Init & MPI_Finalize**: Initializes and finalizes the MPI environment.
        - **MPI_Comm_rank & MPI_Comm_size**: Get the rank (ID) of each process and the total number of processes, respectively.
        - **Matrix Multiplication**: Each process performs matrix multiplication on its part of the data, and we use `MPI` to manage the parallel execution.

        ### 3. Running the Program with Scalasca

        We run the application using Scalasca's `-analyze` option to collect performance data. The application is executed with 4 processes (you can scale this to as many processes as your system allows).

        ### 4. Analyzing the Scalasca Report

        Scalasca generates a performance report after the program finishes execution. We use `scalasca -examine` to examine the collected data, which helps us understand:
        - How much time each process spent in computation vs. communication.
        - Whether there are any bottlenecks in message-passing between processes.
        - Potential improvements to optimize the code for better performance and scalability.

        By examining this profile, we can learn whether there are any inefficiencies in our MPI-based distributed application, such as load imbalance or communication delays.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ## Introduction

        In this lesson, we will profile a distributed MPI application using **Scalasca** on an HPC cluster. Since we are working from a Jupyter notebook, we will not directly execute the commands here, but instead connect to the HPC cluster using SSH and run the profiling steps from there.

        ### What You Will Learn

        - How to connect to an HPC cluster via SSH.
        - How to compile an MPI program with **SCOREP** instrumentation for profiling.
        - How to run an MPI program and collect profiling data using **Scalasca**.
        - How to examine and interpret the performance profiling report.

        ### Requirements

        1. An HPC account with access to a login node.
        2. SSH access to the HPC cluster.
        3. MPI (OpenMPI) and Scalasca installed on the cluster.

        ---

        ## Steps to Run Profiling via SSH

        ### 1. Connect to the HPC Cluster

        To start, you will need to connect to the HPC cluster's login node. Open a terminal (outside of Jupyter) and run the following command to establish an SSH connection:

        ```bash
        ssh <username>@login1.hpcie.labs.faculty.ie.edu

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Replace <username> with your actual username.

        Once connected, you will be in the HPC environment and can proceed with compiling and running your MPI application.

        Use the scratch directory when possible toplay with the system.

        ###2. Load the Necessary Modules
        After logging into the cluster, load the required modules for MPI and Scalasca. Use the following commands:

        ```bash
        module load gcc
        module load openmpi
        module load scalasca
        ```

        This will ensure you have the correct environment for compiling and profiling MPI applications.

        ###3. Write and Compile an MPI Application
        If you haven't already written an MPI application, you can create a simple matrix multiplication MPI program. Create a new .cpp file and write the code:

        ```bash
        nano mpi_matrix_multiply.cpp
        ```

        Paste the following code into the editor:


        ```cpp
        #include <mpi.h>
        #include <iostream>
        #include <vector>

        void matrix_multiply(int N) {
            std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
            std::vector<std::vector<int>> B(N, std::vector<int>(N, 1));
            std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    for (int k = 0; k < N; ++k) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        int main(int argc, char *argv[]) {
            MPI_Init(&argc, &argv);
            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            int N = 1000; // Matrix size
            if (rank == 0) {
                std::cout << "Starting matrix multiplication on " << size << " processes" << std::endl;
            }

            matrix_multiply(N);
    
            std::cout << "Process " << rank << " completed work" << std::endl;
            MPI_Finalize();
            return 0;
        }
        ```

        Now, compile the application with SCOREP instrumentation:

        ```bash
        scorep mpicxx mpi_matrix_multiply.cpp -o mpi_matrix_multiply
        ```

        This command will instrument the code so that Scalasca can profile it.

        ###4. Run the Application with Scalasca
        Once the application is compiled, you can run it using Scalasca to analyze the performance. Run the following command:

        ```bash
        scalasca -analyze mpirun -np 4 -oversubscribe ./mpi_matrix_multiply
        ```

        Here:

        scalasca -analyze is used to collect performance data.
        mpirun -np 4 runs the program with 4 MPI processes.
        ./mpi_matrix_multiply is the executable that was created.

        ###5. Analyze the Profiling Report
        Once the application finishes running, Scalasca will generate a performance report in the form of scorep_* directories. You can analyze this report using:

        ```bash
        export SCOREP_TIMER=gettimeofday

        scalasca -examine --console scorep_*

        ```
        This will generate a detailed analysis of your program's performance, showing you where the bottlenecks are in communication, computation, and load balancing between nodes.

        ###6. Optional: Visualize the Report
        You can visualize the report in various ways using additional tools like cube:

        ```bash
        module load cube
        cube <filename.cubex>
        ```

        This opens a graphical performance analysis report where you can explore function calls, MPI communication patterns, and other insights.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Performance Measurement with PAPI in HPC Applications

        ## Introduction

        In this notebook, we will explore how to use **PAPI** (Performance Application Programming Interface) to measure performance metrics such as floating-point operations in an HPC application. PAPI provides an interface to access hardware performance counters, which allows you to monitor different events at the hardware level (e.g., instructions executed, floating-point operations, cache misses, etc.).

        In this exercise, we will use a simple matrix-vector multiplication example, instrument it with PAPI, and count specific events like **double-precision floating-point operations** (`PAPI_DP_OPS`) and **vector operations** (`PAPI_VEC_DP`).

        ### Why PAPI?

        In High-Performance Computing (HPC), optimizing performance is critical. Tools like PAPI allow developers to get detailed information about how efficiently their applications are using system resources. By understanding where bottlenecks lie (whether in computation or memory access), we can optimize our code to fully exploit the hardware's potential.

        ---

        ## Steps in This Notebook

        1. Write and compile a C program that performs matrix-vector multiplication and integrates PAPI for performance measurement.
        2. Use PAPI to count double-precision floating-point operations (`PAPI_DP_OPS`) and vector operations (`PAPI_VEC_DP`).
        3. Measure and interpret the results.
        4. Test and run the program on an HPC cluster and visualize the collected data.

        Let's dive into the practical part.

        """
    )
    return


app._unparsable_cell(
    r"""
    #Check if papi is available
    !papi_avail
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    #If not installed, install papi library
    !sudo apt-get update
    !sudo apt-get install -y libpapi-dev papi-tools
    !ls /usr/include/papi.h
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Step 1: Write the matrix-vector multiplication program (mvmult_timer.c) with software-based timing
    code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>
    #include <cblas.h>
    #include <time.h>

    void init(int n, double **m, double **v, double **p, int trans) {
        *m = calloc(n*n, sizeof(double));
        *v = calloc(n, sizeof(double));
        *p = calloc(n, sizeof(double));
        for (int i = 0; i < n; i++) {
            (*v)[i] = (i & 1)? -1.0: 1.0;
            if (trans) for (int j = 0; j <= i; j++) (*m)[j*n+i] = 1.0;
            else for (int j = 0; j <= i; j++) (*m)[i*n+j] = 1.0;
        }
    }

    void mult(int size, double *m, double *v, double *p, int trans) {
        int stride = trans? size: 1;
        for (int i = 0; i < size; i++) {
            int mi = trans? i: i*size;
            p[i] = cblas_ddot(size, m+mi, stride, v, 1);
        }
    }

    int main(int argc, char **argv) {
        int n = 1000, trans = 0;
        if (argc > 1) n = strtol(argv[1], NULL, 10);
        if (argc > 2) trans = (argv[2][0] == 't');

        struct timespec start, end;
        double *m, *v, *p;

        clock_gettime(CLOCK_MONOTONIC, &start);
        init(n, &m, &v, &p, trans);
        clock_gettime(CLOCK_MONOTONIC, &end);

        double init_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        clock_gettime(CLOCK_MONOTONIC, &start);
        mult(n, m, v, p, trans);
        clock_gettime(CLOCK_MONOTONIC, &end);

        double mult_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        double s = cblas_dasum(n, p, 1);
        printf(\"Size %d; abs. sum: %f (expected: %d)\\n\", n, s, (n+1)/2);
        printf(\"Timing results (seconds):\\n\");
        printf(\" Initialization time: %f\\n\", init_time);
        printf(\" Multiplication time: %f\\n\", mult_time);

        free(m);
        free(v);
        free(p);

        return 0;
    }
    \"\"\"

    # Write the C code to a file
    with open('mvmult_timer.c', 'w') as f:
        f.write(code)

    # Step 2: Compile the code with CBLAS library
    !gcc -O2 mvmult_timer.c -o mvmult_timer -lcblas

    # Verify that the executable has been created
    !ls -l mvmult_timer
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Step 3: Run the program with a size of 20000
    !./mvmult_timer 20000
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of Results

        The output of the program includes the following key information:

        1. **Matrix Size**: The matrix size used for multiplication (in this case, 20000).
        2. **Absolute Sum of Result**: This verifies that the computation was performed correctly by summing the elements of the result vector.
        3. **PAPI Counts**:
           - **init**: The count of floating-point and vector operations before the matrix-vector multiplication begins.
           - **mult**: The number of floating-point and vector operations that occurred during the matrix-vector multiplication.
           - **sum**: The total count of operations after the computation is complete.

        For example, a typical output might look like this:

        ```plaintext
        Size 20000; abs. sum: 10000.000000 (expected: 10000)
        PAPI counts:
         init: event1: 0               event2: 0
         mult: event1: 804193640        event2: 0
         sum:  event1: 20276           event2: 0

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ---
        ##Stop here, this code needs to be reviewed to adapt it to run in cluster and/or Colab
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # HPC Checkpointing Example using SCR

        ### Introduction

        In high-performance computing (HPC), long-running applications are prone to interruptions such as system failures or maintenance windows. Checkpointing is a technique used to save the state of an application at regular intervals so that it can be resumed from the last saved state in case of failure, avoiding the need to restart the computation from the beginning.

        The **SCR (Scalable Checkpoint/Restart)** library is a lightweight and scalable checkpointing system that allows applications to checkpoint quickly and resume efficiently. It is designed for HPC systems and helps improve application fault tolerance.

        ### Objectives

        In this notebook, we will:
        - Understand the concept of checkpointing and why it is essential in HPC.
        - Learn how to use the SCR library to checkpoint an application.
        - Use MPI to simulate parallel computing with multiple processes.

        ### What Will Be Covered:
        - **Checkpointing**: Save the state of an application to a file at specific intervals.
        - **SCR Library**: Manage checkpoint files efficiently.
        - **MPI Integration**: Coordinate the checkpointing process across multiple processors.

        We will run a simple simulation where each process writes its checkpoint data to a file, which can later be used to restart the application from the checkpoint if needed.

        """
    )
    return


@app.cell
def _():
    # Step 1: Write the checkpointing C code using SCR (mvmult_scr.c)
    code = """
    #include <stdio.h>
    #include <stdlib.h>
    #include "scr.h"
    #include "mpi.h"

    // Function to write a checkpoint
    int write_checkpoint() {
        // Start checkpoint
        SCR_Start_checkpoint();

        // Get the rank of the process
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Create a unique checkpoint file for each rank
        char file[128];
        sprintf(file, "checkpoint/%d_checkpoint.dat", rank);

        // Open the file for writing
        FILE *fp = fopen(file, "w");
        if (fp == NULL) {
            printf("Error: Could not open checkpoint file %s\n", file);
            return -1;
        }

        // Route the file through SCR (optional)
        char scrfile[SCR_MAX_FILENAME];
        SCR_Route_file(file, scrfile);

        // Write checkpoint data to file
        fprintf(fp, "Hello Checkpoint World from process %d\n", rank);
        fclose(fp);

        // Mark the checkpoint as valid
        int valid = 1;
        SCR_Complete_checkpoint(valid);

        return 0;
    }

    int main(int argc, char **argv) {
        // Initialize MPI
        MPI_Init(&argc, &argv);

        // Initialize SCR
        if (SCR_Init() != SCR_SUCCESS) {
            printf("Error: SCR did not initialize\n");
            MPI_Finalize();
            return -1;
        }

        // Simulation loop with checkpointing
        int max_steps = 100;
        for (int step = 0; step < max_steps; step++) {
            // Perform simulation work here...

            // Check if it's time to write a checkpoint
            int checkpoint_flag;
            SCR_Need_checkpoint(&checkpoint_flag);
            if (checkpoint_flag) {
                write_checkpoint();
            }
        }

        // Finalize SCR and MPI
        SCR_Finalize();
        MPI_Finalize();

        return 0;
    }
    """

    # Write the C code to a file
    with open('mvmult_scr.c', 'w') as f:
        f.write(code)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Step 2: Provide Compilation and Running Instructions

        To compile and run this code on an HPC cluster:

        1. Load the necessary modules:
           ```bash
           module load mpi
           module load scr
           ```

        Compile the program:
           ```bash
           mpicc -o mvmult_scr mvmult_scr.c -lscr
           ```
        Run the program with multiple processes:

           ```bash
           mpirun -np 4 ./mvmult_scr
           ```
        This will simulate a simple application where each process saves a checkpoint file with its state. The checkpointing is handled by SCR, and the program can be restarted from the last valid checkpoint
        """
    )
    return


app._unparsable_cell(
    r"""
    !mpicc -o mvmult_scr mvmult_scr.c -lscr
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    !mpirun -np 4 ./mvmult_scr
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ### Part 3: Code Explanation (Markdown)

        ### Main Components of the Checkpointing Code

        1. **MPI Initialization**:
           - The program uses MPI (Message Passing Interface) for parallel execution across multiple processors.
           - `MPI_Init(&argc, &argv)` initializes the MPI environment.
   
        2. **SCR Initialization**:
           - `SCR_Init()` is used to initialize the SCR library. It must be called before any SCR functions can be used.
           - If the initialization fails, the program returns an error.

        3. **Checkpointing Logic**:
           - A simple simulation loop is executed with `max_steps` iterations.
           - On each iteration, the program checks if it's time to perform a checkpoint using `SCR_Need_checkpoint(&checkpoint_flag)`.
           - If a checkpoint is needed, the function `write_checkpoint()` is called, which writes a checkpoint file for each process.

        4. **Writing the Checkpoint**:
           - `SCR_Start_checkpoint()` begins a checkpoint.
           - A unique file is created for each MPI process using its rank (e.g., `checkpoint/0_checkpoint.dat`, `checkpoint/1_checkpoint.dat`).
           - The file is opened and checkpoint data is written, which is a simple message in this case.
           - `SCR_Complete_checkpoint(valid)` marks the checkpoint as valid if it was successful.

        5. **Finalizing SCR and MPI**:
           - `SCR_Finalize()` is called to finalize the SCR library.
           - `MPI_Finalize()` is called to clean up the MPI environment after the computation is finished.

        ### Why Checkpointing is Important in HPC

        - **Fault Tolerance**: Checkpointing allows long-running applications to recover from failures without starting over.
        - **Scalability**: SCR is designed to work efficiently with large-scale parallel applications, minimizing the overhead of checkpointing.
        - **Efficiency**: By checkpointing only when necessary, applications can save resources and avoid frequent restarts.

        This example shows a basic usage of checkpointing in an MPI environment using SCR. The simplicity of the program demonstrates the integration of checkpointing into a larger application and how it can be applied to real-world HPC scenarios.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ---
        ##Stop here, this code needs to be reviewed to adapt it to run in cluster and/or Colab
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Tracing with TAU for Detailed Event Logs in HPC Applications

        In high-performance computing (HPC), tracing provides detailed insights into the dynamic behavior of parallel applications. Unlike profiling, which aggregates performance data, tracing records fine-grained information such as function entries and exits, system calls, and inter-process communication events during program execution.

        **TAU (Tuning and Analysis Utilities)** is a powerful tool for tracing and profiling parallel applications. It supports various parallel programming models, including **MPI** and **OpenMP**. TAU generates trace files that help developers analyze the behavior of their applications in real-time and identify potential bottlenecks in function calls, communication overhead, and load balancing.

        This exercise will show how to use TAU to trace a parallel program. We will:
        1. Compile an MPI-based matrix multiplication program with TAU instrumentation.
        2. Run the program and capture detailed trace logs.
        3. Analyze the trace logs to identify function-level and communication-level details.

        Let’s first create a simple MPI program for matrix multiplication and trace its execution using TAU.

        **Note:** This example assumes you are working in an HPC environment with TAU already installed. We will also load the TAU module, if available.

        """
    )
    return


app._unparsable_cell(
    r"""
    # If using a cluster with modules (e.g., Magi Castle)
    !bash -c \"module load tau/2.30.1 && module list\"

    # TAU is usually not installed via apt-get, so manual installation is required in non-module systems
    """,
    name="_"
)


@app.cell
def _(subprocess):
    def run_tau_tracing():
        # C code for parallel matrix multiplication with MPI
        code = """
        #include <mpi.h>
        #include <stdio.h>
        #include <stdlib.h>

        #define N 4  // Matrix size (N x N)

        void matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    C[i][j] = 0;
                    for (int k = 0; k < N; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }

        int main(int argc, char** argv) {
            MPI_Init(&argc, &argv);  // Initialize MPI

            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            double A[N][N], B[N][N], C[N][N];

            // Initialize matrices A and B
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A[i][j] = rank + i + j;
                    B[i][j] = rank + i - j;
                }
            }

            // Perform matrix multiplication
            matrix_multiply(A, B, C);

            // Print the result matrix from process 0
            if (rank == 0) {
                printf("Matrix C (result):\\n");
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        printf("%f ", C[i][j]);
                    }
                    printf("\\n");
                }
            }

            MPI_Finalize();  // Finalize MPI
            return 0;
        }
        """

        # Save the C code to a file
        with open("mpi_matrix_multiply.c", "w") as file:
            file.write(code)

        # Load the TAU module, if available
        module_check = subprocess.run("module list 2>&1 | grep 'tau'", shell=True, capture_output=True, text=True)
        if module_check.stdout:
            print("TAU module is already loaded.")
        else:
            print("Loading TAU module...")
            subprocess.run("module load tau", shell=True)

        # Compile the C program with TAU instrumentation
        compile_command = "tau_cc -o mpi_matrix_multiply mpi_matrix_multiply.c"
        compile_result = subprocess.run(compile_command, shell=True, capture_output=True, text=True)

        if compile_result.returncode != 0:
            print(f"Compilation failed:\n{compile_result.stderr}")
            return
        else:
            print("Compilation successful.")

        # Run the MPI program with TAU tracing
        tau_run_command = "tau_exec -T mpi -ebs mpirun -np 4 ./mpi_matrix_multiply"
        tau_run_result = subprocess.run(tau_run_command, shell=True, capture_output=True, text=True)

        if tau_run_result.returncode != 0:
            print(f"TAU execution failed:\n{tau_run_result.stderr}")
        else:
            print("TAU tracing complete. Trace files generated.")
            print(tau_run_result.stdout)

    # Run the function to compile and trace the MPI program using TAU
    run_tau_tracing()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## MPI Matrix Multiplication with TAU Profiling

        In this section, we will write, compile, and run an MPI-based matrix multiplication program. We will also use **TAU** for tracing and profiling the performance of this parallel program. TAU provides comprehensive tools for analyzing parallel programs, which can be used to capture function calls, communication patterns, and time spent in different parts of the code.

        ### Steps:
        1. **Write the MPI Matrix Multiplication Code**: This program will perform matrix multiplication using MPI.
        2. **Load TAU Module**: We'll check if the TAU module is available and load it.
        3. **Compile the Code with TAU**: The program will be compiled using `tau_cc` to enable tracing and profiling.
        4. **Run the Program**: We will execute the program with TAU, and collect tracing data.
        5. **View the Profiling Results**: The profiling results will be saved for further analysis.

        Let's begin by writing and saving the C code for the program.

        """
    )
    return


@app.cell
def _(subprocess):
    def run_tau_tracing_1():
        code = '\n    #include <mpi.h>\n    #include <stdio.h>\n    #include <stdlib.h>\n\n    #define N 4  // Matrix size (N x N)\n\n    void matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {\n        for (int i = 0; i < N; i++) {\n            for (int j = 0; j < N; j++) {\n                C[i][j] = 0;\n                for (int k = 0; k < N; k++) {\n                    C[i][j] += A[i][k] * B[k][j];\n                }\n            }\n        }\n    }\n\n    int main(int argc, char** argv) {\n        MPI_Init(&argc, &argv);  // Initialize MPI\n\n        int rank, size;\n        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n        MPI_Comm_size(MPI_COMM_WORLD, &size);\n\n        double A[N][N], B[N][N], C[N][N];\n\n        // Initialize matrices A and B\n        for (int i = 0; i < N; i++) {\n            for (int j = 0; j < N; j++) {\n                A[i][j] = rank + i + j;\n                B[i][j] = rank + i - j;\n            }\n        }\n\n        // Perform matrix multiplication\n        matrix_multiply(A, B, C);\n\n        // Print the result matrix from process 0\n        if (rank == 0) {\n            printf("Matrix C (result):\\n");\n            for (int i = 0; i < N; i++) {\n                for (int j = 0; j < N; j++) {\n                    printf("%f ", C[i][j]);\n                }\n                printf("\\n");\n            }\n        }\n\n        MPI_Finalize();  // Finalize MPI\n        return 0;\n    }\n    '
        with open('mpi_matrix_multiply.c', 'w') as file:
            file.write(code)
        print('MPI Matrix Multiplication code saved as mpi_matrix_multiply.c')
    run_tau_tracing_1()
    load_tau_and_compile = '\n!bash -c "module load tau/2.30.1 && tau_cc -o mpi_matrix_multiply mpi_matrix_multiply.c && mpirun -np 4 ./mpi_matrix_multiply"\n'
    subprocess.run(load_tau_and_compile, shell=True)
    return


@app.cell
def _():
    def run_tau_tracing_2():
        code = '\n    #include <mpi.h>\n    #include <stdio.h>\n    #include <stdlib.h>\n\n    #define N 4  // Matrix size (N x N)\n\n    void matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {\n        for (int i = 0; i < N; i++) {\n            for (int j = 0; j < N; j++) {\n                C[i][j] = 0;\n                for (int k = 0; k < N; k++) {\n                    C[i][j] += A[i][k] * B[k][j];\n                }\n            }\n        }\n    }\n\n    int main(int argc, char** argv) {\n        MPI_Init(&argc, &argv);  // Initialize MPI\n\n        int rank, size;\n        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n        MPI_Comm_size(MPI_COMM_WORLD, &size);\n\n        double A[N][N], B[N][N], C[N][N];\n\n        // Initialize matrices A and B\n        for (int i = 0; i < N; i++) {\n            for (int j = 0; j < N; j++) {\n                A[i][j] = rank + i + j;\n                B[i][j] = rank + i - j;\n            }\n        }\n\n        // Perform matrix multiplication\n        matrix_multiply(A, B, C);\n\n        // Print the result matrix from process 0\n        if (rank == 0) {\n            printf("Matrix C (result):\\n");\n            for (int i = 0; i < N; i++) {\n                for (int j = 0; j < N; j++) {\n                    printf("%f ", C[i][j]);\n                }\n                printf("\\n");\n            }\n        }\n\n        MPI_Finalize();  // Finalize MPI\n        return 0;\n    }\n    '
        with open('mpi_matrix_multiply.c', 'w') as file:
            file.write(code)
        print('MPI Matrix Multiplication code saved as mpi_matrix_multiply.c')
    run_tau_tracing_2()
    return


@app.cell
def _(subprocess):
    load_tau_and_compile_1 = '\n!bash -c "module load tau/2.30.1 && tau_cc -o mpi_matrix_multiply mpi_matrix_multiply.c && mpirun -np 4 ./mpi_matrix_multiply"\n'
    subprocess.run(load_tau_and_compile_1, shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### TAU Tracing Explanation

        In this example, we used **TAU (Tuning and Analysis Utilities)** to trace a parallel matrix multiplication program written in C with **MPI (Message Passing Interface)**. The program was compiled with TAU's instrumentation using the `tau_cc` compiler, and executed with `tau_exec` to capture detailed trace logs.

        Here’s what happened:
        1. **Matrix Multiplication with MPI**: The program multiplies two matrices (4x4 in size) in parallel using multiple MPI processes. Each process computes the product based on its rank.
        2. **TAU Instrumentation**: By using the `tau_cc` compiler and `tau_exec` command, the program was instrumented for tracing. TAU collects detailed information about the program’s execution, including function entries/exits, communication events, and system calls made by each MPI process.
        3. **Trace Files**: TAU generates trace files that can be analyzed using TAU's analysis tools or other visualization tools like ParaProf. These files contain event logs that show the timing and interaction between MPI processes during the execution.

        #### Why Tracing is Important
        Tracing allows developers to gain fine-grained insights into how their programs behave at runtime. In parallel computing, it is often difficult to identify performance bottlenecks just by looking at the source code. Tracing helps by:
        - Providing detailed event logs, showing exactly where the program spends time.
        - Identifying function call hierarchies and how they contribute to total execution time.
        - Highlighting communication patterns and any delays caused by process synchronization or message-passing overhead.
  
        For example, if one MPI process takes significantly longer to complete its work due to poor load balancing, tracing will reveal that process’s timeline, allowing developers to optimize accordingly.

        In this specific case, by analyzing the generated trace files, you can pinpoint where the matrix multiplication operation consumes time and see how the different MPI processes communicate during execution.

        ### What You Should Expect:
        - **Trace Logs**: The logs will include entries showing when the matrix multiplication function was called, how much time each process spent in computation, and how communication was handled.
        - **Visualization Tools**: These trace logs can be visualized using tools like TAU’s **ParaProf** to better understand performance issues such as process imbalance, synchronization delays, and inefficient communication patterns.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sampling Profiling with `perf`

        ### Introduction to Sampling Profiling

        In high-performance computing (HPC), sampling is one of the most common techniques used for profiling applications. Sampling-based profiling periodically checks the state of the program's execution and collects information about which functions are currently running. This method helps developers identify which functions consume the most CPU time, allowing them to focus their optimization efforts on the "hot spots" or areas where the most time is spent.

        In this exercise, we'll be using the `perf` tool, which is widely available on Linux systems, to perform sampling-based profiling of a C++ program. We will write, compile, and run a simple C++ program that processes a large dataset and then use `perf` to collect profiling data and identify which parts of the program are the most time-consuming.

        ### Steps to follow:
        1. **Write and Compile a C++ Program**: We will write a C++ program that processes a vector of data.
        2. **Use `perf` to Profile the Program**: We will run the program with `perf` to collect sampling data and analyze the program's performance.
        3. **Interpret the Results**: After running `perf`, we will interpret the profiling report to identify performance bottlenecks.

        ### Code to Run

        Below is the code that will save, compile, and execute a C++ program to demonstrate sampling profiling. After execution, we will use the `perf` tool to generate a performance report.


        """
    )
    return


app._unparsable_cell(
    r"""
    #]check if perf is installed, if not run the next command to install it
    !perf --version
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # This will install perf. Execute this only if perf is not in the system
    # Update the package list
    !apt-get update
    !apt-cache search linux-tools
    !sudo apt-get install linux-tools-common linux-tools-generic

    !sudo apt-get install git
    !git clone git://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
    !cd linux/tools/perf
    !make
    !sudo cp perf /usr/local/bin/

    # Install perf
    #!sudo apt-get install linux-tools-6.1.85+-6.1.85+ linux-cloud-tools-6.1.85+-6.1.85+
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    !perf --version
    """,
    name="_"
)


@app.cell
def _(subprocess):
    def write_and_compile_cpp():
        # C++ code to process a large vector of data
        code = """
        #include <iostream>
        #include <vector>

        // Function to process data
        void processData(std::vector<int>& data) {
            for (auto& x : data) {
                x *= 2;  // Multiply each element by 2
            }
        }

        int main() {
            std::vector<int> data(1000000, 1);  // Initialize a vector with 1,000,000 elements set to 1
            processData(data);  // Process the data
            std::cout << "Data processed." << std::endl;
            return 0;
        }
        """

        # Save the C++ code to a file
        with open("process_data.cpp", "w") as file:
            file.write(code)

        # Compile the C++ program
        compile_command = "g++ -o process_data process_data.cpp"
        subprocess.run(compile_command, shell=True, check=True)
        print("C++ program compiled successfully.")

    # Write, compile, and run the C++ program
    write_and_compile_cpp()

    # Check if `perf` is available in the system
    try:
        subprocess.run(["perf", "--version"], check=True)
        print("`perf` is available. Running the program with `perf`...")

        # Run the program with `perf` to collect profiling data
        perf_command = "perf record -g ./process_data"
        subprocess.run(perf_command, shell=True, check=True)

        # Generate the `perf` report
        perf_report_command = "perf report"
        subprocess.run(perf_report_command, shell=True)

    except subprocess.CalledProcessError as e:
        print("Error running `perf`:", e)
        print("\nIt seems `perf` is not available or cannot be run in this environment. Please try running these commands manually in a terminal:\n")
        print("1. Compile the program: g++ -o process_data process_data.cpp")
        print("2. Run the program with `perf`: perf record -g ./process_data")
        print("3. View the report: perf report")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ### Explanation of the Code

        1. **C++ Program Overview**:
           The program initializes a large vector (`std::vector<int>`) with 1,000,000 elements, all set to 1. The `processData` function then processes this vector by multiplying each element by 2. Finally, the program prints "Data processed" to the console.

        2. **Profiling with `perf`**:
           - We use the `perf` tool to profile the program and collect sampling data. The command `perf record -g ./process_data` runs the program and collects profiling data, including a call graph to show the function hierarchy.
           - After execution, we run `perf report` to generate a performance report. This report will display how much time was spent in each function.

        3. **Interpreting the Results**:
           - Once the program is run and the profiling data is collected, we use `perf` to view the performance report. It will highlight the time spent in functions such as `main` and `processData`, showing where optimization might be needed.

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

