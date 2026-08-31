import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Hybrid MPI + OpenMP + CUDA Example: Solving Large System of Linear Equations

        In this notebook, we will explore a hybrid parallel programming example. The problem we're solving is a large system of linear equations. To do this, we'll use a combination of **MPI**, **OpenMP**, and **CUDA**.

        ## Problem

        The system of linear equations can be represented in matrix form as:

        \[ A \times x = b \]

        Where:
        - `A` is a matrix of coefficients,
        - `x` is the vector of unknowns,
        - `b` is the known vector of results.

        The goal is to parallelize this process across multiple **nodes** using **MPI**, within each node using **OpenMP**, and to accelerate matrix operations using **CUDA** on the GPU.

        ### Steps:
        1. **Domain Decomposition with MPI**: Split the problem across multiple nodes.
        2. **Multi-threading with OpenMP**: Perform parallel matrix calculations within each node.
        3. **GPU Acceleration with CUDA**: Offload intensive matrix operations to the GPU.

        We will first initialize MPI, then allocate memory and perform calculations with OpenMP. CUDA will handle the heavy matrix multiplications.

        ---

        ### MPI Setup

        We start by initializing MPI, which will allow communication between different nodes. MPI will also handle splitting the work between different processors.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Install OpenMPI
    !apt-get update
    !apt-get install -y openmpi-bin openmpi-common libopenmpi-dev

    # Verify CUDA installation
    !nvcc --version

    # ** Locate MPI Headers and Libraries**
    # Find mpi.h
    mpi_h_paths = !find /usr -name mpi.h
    print(\"MPI Header Paths:\")
    for path in mpi_h_paths:
        print(path)

    # Find libmpi.so
    mpi_lib_paths = !find /usr -name libmpi.so
    print(\"\nMPI Library Paths:\")
    for path in mpi_lib_paths:
        print(path)
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # ** Write the Updated C Code**
    cuda_mpi_code = \"\"\"
    #include <mpi.h>
    #include <omp.h>
    #include <cuda_runtime.h>
    #include <stdio.h>
    #include <stdlib.h>

    // CUDA Kernel for matrix multiplication
    __global__ void gpu_matrix_mult(double *A, double *B, double *C, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < N && col < N) {
            double sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    int main(int argc, char* argv[]) {
        // Initialize MPI
        MPI_Init(&argc, &argv);

        int world_size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int N = 1024;
        double *A, *B, *C;

        // Allocate memory for matrices
        A = (double*)malloc(N * N * sizeof(double));
        B = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));

        // Initialize matrices in parallel with OpenMP
        #pragma omp parallel for
        for (int i = 0; i < N * N; i++) {
            A[i] = rand() / (double)RAND_MAX;
            B[i] = rand() / (double)RAND_MAX;
        }

        // Allocate device memory for CUDA
        double *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, N * N * sizeof(double));
        cudaMalloc((void **)&d_B, N * N * sizeof(double));
        cudaMalloc((void **)&d_C, N * N * sizeof(double));

        // Copy data to GPU
        cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);

        // CUDA kernel configuration
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch CUDA kernel for matrix multiplication
        gpu_matrix_mult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        // Wait for GPU to finish
        cudaDeviceSynchronize();

        // Copy result back to host
        cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

        // (Optional) Verify a few results
        if(rank == 0) {
            printf(\"C[0] = %f\\n\", C[0]);
            printf(\"C[N*N-1] = %f\\n\", C[N*N-1]);
        }

        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(A);
        free(B);
        free(C);

        // Finalize MPI
        MPI_Finalize();
        return 0;
    }
    \"\"\"

    with open(\"hybrid_mpi_openmp_cuda.cu\", \"w\") as cuda_file:
        cuda_file.write(cuda_mpi_code)

    # **Cell 5: Compile the CUDA-MPI Program**
    # Automatically detect MPI include and library paths using mpicc
    # Get MPI compile flags (includes)
    mpi_cflags = !mpicc --showme:compile
    # Get MPI link flags (libraries)
    mpi_libs = !mpicc --showme:link

    # Combine the flags into strings
    mpi_cflags = \" \".join(mpi_cflags)
    mpi_libs = \" \".join(mpi_libs)

    # Compile the CUDA-MPI program with dynamic flags
    compile_command = f\"nvcc hybrid_mpi_openmp_cuda.cu -o cuda_mpi -Xcompiler \\"-fopenmp {mpi_cflags}\\" {mpi_libs}\"
    print(\"Compilation Command:\")
    print(compile_command)
    !{compile_command}

    # **Cell 6: Set Environment Variables for MPI**
    import os
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # **Cell 7: Run the CUDA-MPI Program**
    !mpirun --allow-run-as-root --oversubscribe -np 2 ./cuda_mpi
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        ---

        ### Markdown Explanation of the Provided Code

        Here's the markdown explanation for the code you've provided:

        ```markdown
        # Explanation of the MPI-OpenMP-CUDA Code

        The code below is a hybrid MPI-OpenMP-CUDA program designed for distributed matrix multiplication using both CPU and GPU resources. Here's a breakdown of each part of the code:

        1. **CUDA Kernel for Matrix Multiplication**  
           The `gpu_matrix_mult` function is a CUDA kernel that performs matrix multiplication on the GPU. It computes a single element of the result matrix `C` by multiplying the corresponding rows and columns of matrices `A` and `B`. The kernel uses 2D thread and block indices to map the computation to the correct element in the matrix.

        2. **MPI Initialization**  
           The program starts by initializing MPI with `MPI_Init`, obtaining the total number of MPI processes (`world_size`) and the rank of the current process (`rank`). This is necessary to manage distributed execution across multiple nodes or processors.

        3. **Memory Allocation and Initialization**  
           Matrices `A`, `B`, and `C` are allocated in the host (CPU) memory using `malloc`. These matrices are initialized with random values in parallel using OpenMP's `#pragma omp parallel for` directive. This parallelizes the initialization process to utilize multiple CPU cores.

        4. **CUDA Memory Allocation and Data Transfer**  
           Memory for matrices `A`, `B`, and `C` is allocated on the GPU using `cudaMalloc`. The data for matrices `A` and `B` is then copied from the host memory to the GPU using `cudaMemcpy`.

        5. **Launching the CUDA Kernel**  
           The CUDA kernel `gpu_matrix_mult` is launched to compute the result matrix `C` on the GPU. The grid and block dimensions are configured using `dim3 threadsPerBlock(16, 16)` and `dim3 blocksPerGrid`, which ensures that the entire matrix is processed in parallel by the GPU threads.

        6. **Synchronization and Data Transfer Back to Host**  
           After the GPU finishes executing the kernel, the program synchronizes with `cudaDeviceSynchronize`. The result matrix `C` is copied back from the GPU to the host using `cudaMemcpy`.

        7. **Result Verification**  
           For debugging purposes, the program prints two elements from the result matrix `C` (the first and last elements). This is done only by the MPI process with rank 0 to avoid duplicate outputs from multiple processes.

        8. **Cleanup**  
           Once the computation is done, the allocated memory on both the host and device is freed using `free` and `cudaFree` respectively. MPI is finalized with `MPI_Finalize`.

        9. **Compilation and Execution**  
           The program is compiled using `nvcc` with appropriate MPI and OpenMP flags. It is then executed using `mpirun` with multiple processes, utilizing both distributed computing (MPI) and parallelism (OpenMP and CUDA).

        By working through this code, you will learn how to effectively combine MPI for distributed execution, OpenMP for multi-threading on the CPU, and CUDA for GPU acceleration.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise: Modifying the MPI-OpenMP-CUDA Code

        In this exercise, you will modify the existing MPI-OpenMP-CUDA hybrid code to further explore how distributed computing works in combination with GPU acceleration. Follow the steps below:

        1. **Task 1: Add Output for Verification**  
           Modify the code to print more elements from the result matrix `C`. Currently, the code only prints `C[0]` and `C[N*N-1]`.  
           - Add additional print statements to display elements like `C[N/2]` and `C[N*N/2]`. This will help verify that the matrix multiplication is computed correctly across more elements.

        2. **Task 2: Distribute Matrix Computation with MPI**  
           Currently, all MPI processes are doing the same work. Modify the code so that each MPI process computes a different section of the matrix `C`.  
           - For example, divide the matrix `C` into two parts, where the first MPI process computes the first half and the second MPI process computes the second half. This will involve adjusting the indices for each MPI rank.

        3. **Task 3: Experiment with OpenMP Threads**  
           Adjust the number of OpenMP threads used in the code. Use the `OMP_NUM_THREADS` environment variable to experiment with different thread counts, and observe how it affects performance.

        Make sure to test your changes by running the program with different numbers of MPI processes and OpenMP threads. Discuss in your report how each modification affects the performance and correctness of the program.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Solution to the Exercise: Modifying the MPI-OpenMP-CUDA Code

        1. **Task 1: Add Output for Verification**

           To print more elements from the result matrix `C`, you can modify the `if(rank == 0)` block to include additional print statements. Here's an example:

           ```c
           if(rank == 0) {
               printf("C[0] = %f\n", C[0]);
               printf("C[N/2] = %f\n", C[N/2]);
               printf("C[N*N/2] = %f\n", C[N*N/2]);
               printf("C[N*N-1] = %f\n", C[N*N-1]);
           }

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OpenACC MPI Hybrid Code

        This version of the code leverages **OpenACC** to offload matrix multiplication computations to a GPU, replacing the CUDA-specific implementation. OpenACC is designed to make parallel programming easier by allowing developers to write portable code that can run on CPUs and GPUs without needing to manage low-level GPU specifics like memory allocation or kernel launches. Here's a breakdown of the code:

        ### 1. MPI Initialization
        The program initializes MPI with the following commands:
        ```c
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Init initializes the MPI environment, and MPI_Comm_size and MPI_Comm_rank retrieve the number of processes and the rank of the current process. This is important for distributed memory systems.
        2. Memory Allocation
        Memory for the matrices A, B, and C is dynamically allocated on the host (CPU):

        c
        Copy code
        A = (double*)malloc(N * N * sizeof(double));
        B = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));
        These matrices are later used for matrix multiplication, and OpenACC will handle their offloading to the GPU.

        3. Matrix Initialization
        The matrices are initialized using OpenMP on the CPU:

        c
        Copy code
        #pragma omp parallel for
        for (int i = 0; i < N * N; i++) {
            A[i] = rand() / (double)RAND_MAX;
            B[i] = rand() / (double)RAND_MAX;
        }
        This allows the initialization of the matrices to take advantage of multi-threading on the CPU before the computation is offloaded to the GPU.

        4. GPU Offloading with OpenACC
        The matrix multiplication is offloaded to the GPU using the OpenACC directive:

        c
        Copy code
        #pragma acc parallel loop collapse(2) copyin(A[0:N*N], B[0:N*N]) copyout(C[0:N*N])
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < N; ++col) {
                double sum = 0;
                for (int i = 0; N; ++i) {
                    sum += A[row * N + i] * B[i * N + col];
                }
                C[row * N + col] = sum;
            }
        }
        #pragma acc parallel loop: This directive tells the compiler to parallelize the loop and offload it to the GPU.
        collapse(2): Combines the two outer loops (row and col) into a single loop for better parallelization.
        copyin(A[0:N*N], B[0:N*N]): This copies the matrices A and B from the CPU memory to the GPU memory.
        copyout(C[0:N*N]): This copies the result matrix C from the GPU back to the CPU after the computation.
        5. Optional Verification
        The program prints the values of some elements in matrix C for verification. This is done by the process with rank 0:

        c
        Copy code
        if(rank == 0) {
            printf("C[0] = %f\n", C[0]);
            printf("C[N*N-1] = %f\n", C[N*N-1]);
        }
        6. Memory Cleanup
        After the matrix multiplication is complete, the dynamically allocated memory for the matrices is freed:

        c
        Copy code
        free(A); free(B); free(C);
        Finally, MPI is finalized with MPI_Finalize(), which terminates the MPI environment.
        """
    )
    return


app._unparsable_cell(
    r"""
    # ** Write the  OpenACC-MPI Code**
    openacc_mpi_code = \"\"\"
    #include <mpi.h>
    #include <openacc.h>
    #include <stdio.h>
    #include <stdlib.h>

    // OpenACC kernel for matrix multiplication
    void gpu_matrix_mult(double *A, double *B, double *C, int N) {
        #pragma acc parallel loop collapse(2) copyin(A[0:N*N], B[0:N*N]) copyout(C[0:N*N])
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < N; ++col) {
                double sum = 0;
                for (int i = 0; i < N; ++i) {
                    sum += A[row * N + i] * B[i * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }

    int main(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);

        int world_size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int N = 1024;
        double *A, *B, *C;

        A = (double*)malloc(N * N * sizeof(double));
        B = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));

        #pragma omp parallel for
        for (int i = 0; i < N * N; i++) {
            A[i] = rand() / (double)RAND_MAX;
            B[i] = rand() / (double)RAND_MAX;
        }

        gpu_matrix_mult(A, B, C, N);

        if(rank == 0) {
            printf(\"C[0] = %f\\n\", C[0]);
            printf(\"C[N*N-1] = %f\\n\", C[N*N-1]);
        }

        free(A); free(B); free(C);

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Save the OpenACC-MPI code to a file
    with open(\"openacc_mpi.c\", \"w\") as openacc_file:
        openacc_file.write(openacc_mpi_code)
    # Get MPI compile flags (includes)
    mpi_cflags = !mpicc --showme:compile
    # Get MPI link flags (libraries)
    mpi_libs = !mpicc --showme:link

    # Combine the flags into strings
    mpi_cflags = \" \".join(mpi_cflags)
    mpi_libs = \" \".join(mpi_libs)

    # Compile the OpenACC-MPI program with dynamic flags
    compile_command = f\"mpicc -fopenacc openacc_mpi.c -o openacc_mpi {mpi_cflags} {mpi_libs}\"
    print(\"Compilation Command:\")
    print(compile_command)
    !{compile_command}

    # Set the environment variables required by MPI
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    # Run the compiled OpenACC-MPI program
    !mpirun --allow-run-as-root --oversubscribe -np 2 ./openacc_mpi
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OpenMP Offload Version of MPI-OpenMP Hybrid Code

        This modified version of the code uses **OpenMP Offload** instead of CUDA for GPU acceleration. OpenMP Offload allows you to offload computational work to an available GPU or accelerator using OpenMP directives. Here's a breakdown of the key changes:

        1. **OpenMP Offload for GPU**  
           The `gpu_matrix_mult` function, which previously used a CUDA kernel, now uses OpenMP's offload features. The `#pragma omp target teams distribute parallel for` directive offloads the computation to the GPU:
           ```cpp
           #pragma omp target teams distribute parallel for collapse(2) map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
           for (int row = 0; row < N; ++row) {
               for (int col = 0; col < N; ++col) {
                   double sum = 0;
                   for (int i = 0; i < N; ++i) {
                       sum += A[row * N + i] * B[i * N + col];
                   }
                   C[row * N + col] = sum;
               }
           }

        """
    )
    return


@app.cell
def _():
    # ** Write the Updated OpenMP Offload Code**
    openmp_mpi_code = """
    #include <mpi.h>
    #include <omp.h>
    #include <stdio.h>
    #include <stdlib.h>

    // OpenMP Offload kernel for matrix multiplication
    void gpu_matrix_mult(double *A, double *B, double *C, int N) {
        #pragma omp target teams distribute parallel for collapse(2) map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < N; ++col) {
                double sum = 0;
                for (int i = 0; i < N; ++i) {
                    sum += A[row * N + i] * B[i * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }

    int main(int argc, char* argv[]) {
        // Initialize MPI
        MPI_Init(&argc, &argv);

        int world_size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int N = 1024;
        double *A, *B, *C;

        // Allocate memory for matrices
        A = (double*)malloc(N * N * sizeof(double));
        B = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));

        // Initialize matrices in parallel with OpenMP
        #pragma omp parallel for
        for (int i = 0; i < N * N; i++) {
            A[i] = rand() / (double)RAND_MAX;
            B[i] = rand() / (double)RAND_MAX;
        }

        // Offload matrix multiplication to GPU
        gpu_matrix_mult(A, B, C, N);

        // (Optional) Verify a few results
        if(rank == 0) {
            printf("C[0] = %f\\n", C[0]);
            printf("C[N*N-1] = %f\\n", C[N*N-1]);
        }

        // Clean up
        free(A);
        free(B);
        free(C);

        // Finalize MPI
        MPI_Finalize();
        return 0;
    }
    """

    # Save the OpenMP Offload code to a file
    with open("openmp_mpi_offload.c", "w") as openmp_file:
        openmp_file.write(openmp_mpi_code)
    return


app._unparsable_cell(
    r"""
    # Get MPI compile flags (includes)
    mpi_cflags = !mpicc --showme:compile
    # Get MPI link flags (libraries)
    mpi_libs = !mpicc --showme:link

    # Combine the flags into strings
    mpi_cflags = \" \".join(mpi_cflags)
    mpi_libs = \" \".join(mpi_libs)

    # Compile the OpenMP-MPI program with dynamic flags
    compile_command = f\"mpicc openmp_mpi_offload.c -o openmp_mpi_offload -fopenmp {mpi_cflags} {mpi_libs}\"
    print(\"Compilation Command:\")
    print(compile_command)
    !{compile_command}
    # Set the environment variables required by OpenMP and MPI
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

    # Set the number of OpenMP threads (optional)
    os.environ['OMP_NUM_THREADS'] = '4'
    # Run the compiled OpenMP-MPI program
    !mpirun --allow-run-as-root --oversubscribe -np 2 ./openmp_mpi_offload
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Step-by-Step Guide: Converting a Serial Program into a Hybrid Application

        This section provides a practical guide on converting a serial 2D heat conduction simulation into a hybrid parallel application using MPI, OpenMP, and GPU acceleration. By breaking down the process step-by-step, this guide aims to help students understand how to apply hybrid computing techniques in real-world scenarios, optimizing performance and scalability.

        ## Introduction to the Problem: Parallelizing a 2D Heat Conduction Simulation

        Simulating heat distribution over a two-dimensional plate is a fundamental problem in computational physics and engineering. It involves solving the heat equation—a partial differential equation that describes how heat diffuses through a given region over time. This problem has real-world applications in material science, thermodynamics, and mechanical engineering.

        ### Problem Overview:
        Imagine a metal plate with fixed temperatures along its edges and an initial temperature distribution within. Over time, heat will flow from hotter regions to cooler ones until the plate reaches thermal equilibrium. Our goal is to model this process by calculating the temperature at each point on the plate at successive time steps.

        ### Limitations of the Serial Implementation:
        - **Long Execution Time**: Large grid sizes and many time steps result in long execution times.
        - **Memory Constraints**: Large simulations may exceed the memory capacity of a single machine.
        - **Inefficiency for Real-World Applications**: Serial approaches are inefficient for real-time simulations or high-resolution grids.

        To overcome these limitations, we'll parallelize the simulation using a hybrid parallelization strategy involving MPI for distributed memory parallelism, OpenMP for multi-threading, and GPU acceleration for compute-intensive tasks.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Serial Version: 2D Heat Conduction Simulation

        In this section, we provide the serial version of the 2D heat conduction simulation. This code calculates the temperature distribution over a 2D plate using the finite difference method. It runs on a single CPU core without any parallelization, and can be used as a baseline for comparing the performance improvements achieved with MPI, OpenMP, and CUDA in the parallel versions.

        ## Serial Code in C



        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explanation of the Serial 2D Heat Conduction Code

        This serial version of the 2D heat conduction simulation models the heat distribution over a plate using a finite difference method. The temperature at each point on the plate is updated iteratively based on the temperatures of its neighboring points.

        ## Key Components of the Code:

        ### 1. Grid Initialization (`initialize()` function):
        - The grid (or 2D array `temp`) represents the temperature at each point on the plate.
        - **Boundary Conditions**: The temperature at the edges of the grid is fixed at 100°C, representing a heated boundary. The interior of the grid is initialized to 0°C.
            ```c
            if (i == 0 || i == NY-1 || j == 0 || j == NX-1) {
                temp[i][j] = 100.0; // Hot edges
            }
            ```

        ### 2. Temperature Update (`update()` function):
        - The **finite difference method (FDM)** is used to compute the temperature at each interior grid point based on the temperatures of its neighbors.
            ```c
            temp_new[i][j] = temp[i][j] + alpha * (
                (temp[i+1][j] - 2 * temp[i][j] + temp[i-1][j]) / (dx * dx) +
                (temp[i][j+1] - 2 * temp[i][j] + temp[i][j-1]) / (dy * dy)
            );
            ```
        - The constant `alpha` represents the thermal diffusivity, and `dx` and `dy` represent the grid spacing.
        - **Time-stepping loop**: The temperature is updated for `NSTEPS` iterations, simulating the heat diffusion over time.

        ### 3. Computational Limitations:
        - **Long Execution Time**: For large grids (e.g., `1000x1000`) and many time steps, this serial version can take a long time to run.
        - **Memory Usage**: A large grid requires significant memory, which could be a limitation for larger simulations.
        - **No Parallelism**: The serial version does not exploit multi-core CPUs or GPUs, making it inefficient for larger or more complex simulations.

        ---

        ## Next Steps:
        - The serial code provides a baseline for performance. We can measure the execution time of this version and compare it against parallel versions using MPI, OpenMP, and CUDA to see the performance gains from parallelism.

        """
    )
    return


@app.cell
def _():
    # This script performs the following:
    # 1. Writes and compiles a serial C program for 2D heat conduction.
    # 2. Writes and compiles an OpenMP-parallelized version of the same program.
    # 3. Runs both programs, captures their execution times.
    # 4. Compares the execution times and calculates speedup and efficiency.

    import subprocess
    import sys
    import re

    # Function to write C code to a file
    def write_c_code(filename, code):
        with open(filename, "w") as f:
            f.write(code)

    # Function to compile C code
    def compile_c_code(source, output, flags=[]):
        compile_command = ["gcc"] + flags + [source, "-o", output]
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error compiling {source}:\n{result.stderr}")
            sys.exit(1)
        else:
            print(f"Compiled {source} successfully.")

    # Function to run executable and capture output
    def run_executable(exec_path):
        result = subprocess.run([f"./{exec_path}"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {exec_path}:\n{result.stderr}")
            sys.exit(1)
        else:
            return result.stdout.strip()

    # 1. Write Serial C Code
    serial_code = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>

    #define NX 3000    // Number of grid points in X-direction
    #define NY 3000    // Number of grid points in Y-direction
    #define NSTEPS 500 // Number of time steps

    double temp[NY][NX];
    double temp_new[NY][NX];

    void initialize() {
        for (int i = 0; i < NY; i++) {
            for (int j = 0; j < NX; j++) {
                temp[i][j] = 0.0; // Initial temperature
                // Set boundary conditions
                if (i == 0 || i == NY - 1 || j == 0 || j == NX - 1) {
                    temp[i][j] = 100.0; // Hot edges
                }
            }
        }
    }

    void update() {
        double alpha = 0.01; // Thermal diffusivity
        double dx = 1.0;
        double dy = 1.0;

        for (int step = 0; step < NSTEPS; step++) {
            for (int i = 1; i < NY - 1; i++) {
                for (int j = 1; j < NX - 1; j++) {
                    temp_new[i][j] = temp[i][j] + alpha * (
                        (temp[i+1][j] - 2 * temp[i][j] + temp[i-1][j]) / (dx * dx) +
                        (temp[i][j+1] - 2 * temp[i][j] + temp[i][j-1]) / (dy * dy)
                    );
                }
            }
            // Copy temp_new to temp for the next iteration
            for (int i = 1; i < NY - 1; i++) {
                for (int j = 1; j < NX - 1; j++) {
                    temp[i][j] = temp_new[i][j];
                }
            }
        }
    }

    int main() {
        clock_t start, end;
        double cpu_time_used;

        initialize();

        start = clock();
        update();
        end = clock();

        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Serial Execution Time: %f seconds\\n", cpu_time_used);

        return 0;
    }
    """

    serial_filename = "heat_serial.c"
    write_c_code(serial_filename, serial_code)
    print("Serial C code written to heat_serial.c")

    # 2. Compile Serial Code
    compile_c_code(serial_filename, "heat_serial")
    print()

    # 3. Run Serial Executable
    print("Running Serial Executable...")
    serial_output = run_executable("heat_serial")
    print(serial_output)
    print()

    # Extract Serial Execution Time using regex
    serial_time_match = re.search(r"Serial Execution Time:\s+([0-9.]+)\s+seconds", serial_output)
    if serial_time_match:
        serial_time = float(serial_time_match.group(1))
    else:
        print("Failed to extract serial execution time.")
        sys.exit(1)

    # 4. Write OpenMP C Code
    openmp_code = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    #include <omp.h>

    #define NX 1000    // Number of grid points in X-direction
    #define NY 1000    // Number of grid points in Y-direction
    #define NSTEPS 500 // Number of time steps

    double temp[NY][NX];
    double temp_new[NY][NX];

    void initialize() {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NY; i++) {
            for (int j = 0; j < NX; j++) {
                temp[i][j] = 0.0; // Initial temperature
                // Set boundary conditions
                if (i == 0 || i == NY - 1 || j == 0 || j == NX - 1) {
                    temp[i][j] = 100.0; // Hot edges
                }
            }
        }
    }

    void update() {
        double alpha = 0.01; // Thermal diffusivity
        double dx = 1.0;
        double dy = 1.0;

        for (int step = 0; step < NSTEPS; step++) {
            #pragma omp parallel for collapse(2)
            for (int i = 1; i < NY - 1; i++) {
                for (int j = 1; j < NX - 1; j++) {
                    temp_new[i][j] = temp[i][j] + alpha * (
                        (temp[i+1][j] - 2 * temp[i][j] + temp[i-1][j]) / (dx * dx) +
                        (temp[i][j+1] - 2 * temp[i][j] + temp[i][j-1]) / (dy * dy)
                    );
                }
            }
            // Copy temp_new to temp for the next iteration
            #pragma omp parallel for collapse(2)
            for (int i = 1; i < NY - 1; i++) {
                for (int j = 1; j < NX - 1; j++) {
                    temp[i][j] = temp_new[i][j];
                }
            }
        }
    }

    int main() {
        clock_t start, end;
        double cpu_time_used;
        int num_threads = 2; // You can adjust the number of threads

        omp_set_num_threads(num_threads);

        initialize();

        start = clock();
        update();
        end = clock();

        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("OpenMP Execution Time with %d threads: %f seconds\\n", num_threads, cpu_time_used);

        return 0;
    }
    """

    openmp_filename = "heat_openmp.c"
    write_c_code(openmp_filename, openmp_code)
    print("OpenMP C code written to heat_openmp.c")

    # 5. Compile OpenMP Code
    compile_c_code(openmp_filename, "heat_openmp", flags=["-fopenmp"])
    print()

    # 6. Run OpenMP Executable
    print("Running OpenMP Executable...")
    openmp_output = run_executable("heat_openmp")
    print(openmp_output)
    print()

    # Extract OpenMP Execution Time and Number of Threads using regex
    openmp_match = re.search(r"OpenMP Execution Time with (\d+) threads:\s+([0-9.]+)\s+seconds", openmp_output)
    if openmp_match:
        openmp_threads = int(openmp_match.group(1))
        openmp_time = float(openmp_match.group(2))
    else:
        print("Failed to extract OpenMP execution time or number of threads.")
        sys.exit(1)

    # 7. Compare and Print Results
    print("--- Execution Time Comparison ---")
    print(f"Serial Execution Time: {serial_time:.6f} seconds")
    print(f"OpenMP Execution Time with {openmp_threads} threads: {openmp_time:.6f} seconds")

    # Calculate Speedup and Efficiency
    speedup = serial_time / openmp_time if openmp_time > 0 else float('inf')
    efficiency = (speedup / openmp_threads) * 100 if openmp_threads > 0 else 0

    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.2f}%")
    print()

    # 8. Analyze Performance Discrepancy
    if openmp_time > serial_time:
        print("**Observation:** The OpenMP version is slower than the serial version.")
        print("**Possible Reasons:**")
        print("- Overhead from thread creation and synchronization.")
        print("- Limited number of physical CPU cores in the Colab environment.")
        print("- Inefficient parallelization or cache contention.")
        print("- The problem size may not be large enough to benefit from parallelization.")
        print("- OpenMP directives might not be optimally placed.")
        print()
        print("**Recommendations:**")
        print("- Increase the problem size (e.g., larger grid or more time steps) to better utilize parallelism.")
        print("- Experiment with different numbers of threads to find the optimal count.")
        print("- Optimize OpenMP directives, such as using appropriate scheduling strategies.")
        print("- Profile the code to identify and address bottlenecks.")
    else:
        print("**Observation:** The OpenMP version is faster than the serial version.")
        print("**Performance Benefits Achieved Through OpenMP Parallelization.**")
    return re, subprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code Explanation: Parallelizing 2D Heat Conduction with MPI, OpenMP, and CUDA

        ## 1. **MPI for Distributed Memory Parallelism**
        - **Domain Decomposition**: The 2D grid is split across MPI processes. Each process manages a subset of rows (`local_NY`) to parallelize computation.
        - **MPI Initialization**:
            ```c
            MPI_Init(&argc, &argv);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            ```
            These functions initialize the MPI environment and determine the rank and number of processes.
  
        - **Data Partitioning**: Each process works on a portion of the grid. The grid is divided along the Y-dimension, and each process computes a subset of rows.

        ## 2. **OpenMP for Intra-node Parallelism**
        - **OpenMP Directives**: OpenMP is used within each MPI process for initializing the grid and performing the update in parallel. OpenMP ensures that loops over grid points are distributed across multiple CPU threads.

        ## 3. **CUDA for GPU Acceleration**
        - **GPU Offloading**: The temperature update step is offloaded to the GPU using a CUDA kernel. The kernel computes the new temperatures in parallel across grid points.
            ```c
            __global__ void update_kernel(double *temp, double *temp_new, int NX, int NY, double alpha);
            ```

        - **Kernel Execution**: The kernel is launched with a 2D grid and block configuration using:
            ```c
            dim3 blockDim(16, 16);
            dim3 gridDim((NX + blockDim.x - 1) / blockDim.x, (local_NY + blockDim.y - 1) / blockDim.y);
            update_kernel<<<gridDim, blockDim>>>(d_temp, d_temp_new, NX, local_NY, alpha);
            ```

        ## 4. **MPI Communication**
        - Each process exchanges boundary rows with neighboring processes to ensure the boundary conditions are correctly handled across the grid. In this simplified version, we do not show the MPI communication for boundary exchange, but it can be added with `MPI_Isend` and `MPI_Irecv` for non-blocking communication.

        ## 5. **Hybrid Approach**
        - The combination of **MPI for inter-node communication**, **OpenMP for intra-node parallelism**, and **CUDA for GPU acceleration** ensures efficient utilization of hardware resources, enabling scalable performance for large simulations.

        ### Key Concepts:
        - **Distributed Memory Parallelism**: MPI splits the grid across multiple nodes, reducing the memory load on a single machine.
        - **Shared Memory Parallelism**: OpenMP ensures that CPU cores within a node work together efficiently.
        - **GPU Acceleration**: CUDA accelerates the most compute-intensive parts of the simulation, providing massive parallelism.

        """
    )
    return


app._unparsable_cell(
    r"""
    # 1. Installs MPI and verifies CUDA installation.
    # ------------------------------
    # 1. Install MPI and Verify CUDA
    # ------------------------------
    print(\"Installing MPI...\")
    # Update package lists and install MPI
    !apt-get update -y
    !apt-get install -y mpi-default-bin mpi-default-dev

    print(\"\nVerifying MPI installation:\")
    # Check MPI compiler version
    !mpicc --version

    print(\"\nVerifying CUDA installation:\")
    # Check CUDA compiler version
    !nvcc --version
    """,
    name="_"
)


@app.cell
def _(os, re, subprocess):
    def write_code(filename, code):
        with open(filename, 'w') as f:
            f.write(code)
        print(f'Code written to {filename}')

    def compile_c(source, output, flags=[]):
        compile_command = ['gcc'] + flags + [source, '-o', output]
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'Error compiling {source}:\n{result.stderr}')
            return False
        else:
            print(f'Compiled {source} successfully.')
            return True

    def compile_mpi_openmp(source, output, flags=[]):
        compile_command = ['mpicc'] + flags + [source, '-o', output]
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'Error compiling {source}:\n{result.stderr}')
            return False
        else:
            print(f'Compiled {source} successfully.')
            return True

    def compile_mpi_openmp_cuda(source, output, flags=[]):
        mpi_include = '/usr/include/openmpi'
        mpi_lib = '/usr/lib/x86_64-linux-gnu'
        if not os.path.exists(f'{mpi_include}/mpi.h'):
            mpi_h_path = subprocess.getoutput('find /usr/include -name mpi.h')
            if mpi_h_path:
                mpi_include = os.path.dirname(mpi_h_path)
                print(f'Found mpi.h at {mpi_h_path}')
            else:
                print('mpi.h not found. Cannot compile CUDA program with MPI.')
                return False
        compile_command = ['nvcc', '-Xcompiler', '-fopenmp', source, '-o', output, f'-I{mpi_include}', f'-L{mpi_lib}', '-lmpi']
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'Error compiling {source}:\n{result.stderr}')
            return False
        else:
            print(f'Compiled {source} successfully.')
            return True

    def run_executable_1(exec_path):
        try:
            result = subprocess.run([f'./{exec_path}'], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f'Error running {exec_path}:\n{e.output}')
            return None
    serial_code_1 = '\n#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>\n\n#define NX 3000    // Number of grid points in X-direction\n#define NY 3000    // Number of grid points in Y-direction\n#define NSTEPS 500 // Number of time steps\n\ndouble temp[NY][NX];\ndouble temp_new[NY][NX];\n\nvoid initialize() {\n    for (int i = 0; i < NY; i++) {\n        for (int j = 0; j < NX; j++) {\n            temp[i][j] = 0.0; // Initial temperature\n            // Set boundary conditions\n            if (i == 0 || i == NY - 1 || j == 0 || j == NX - 1) {\n                temp[i][j] = 100.0; // Hot edges\n            }\n        }\n    }\n}\n\nvoid update() {\n    double alpha = 0.01; // Thermal diffusivity\n    double dx = 1.0;\n    double dy = 1.0;\n\n    for (int step = 0; step < NSTEPS; step++) {\n        for (int i = 1; i < NY - 1; i++) {\n            for (int j = 1; j < NX - 1; j++) {\n                temp_new[i][j] = temp[i][j] + alpha * (\n                    (temp[i+1][j] - 2 * temp[i][j] + temp[i-1][j]) / (dx * dx) +\n                    (temp[i][j+1] - 2 * temp[i][j] + temp[i][j-1]) / (dy * dy)\n                );\n            }\n        }\n        // Copy temp_new to temp for the next iteration\n        for (int i = 1; i < NY - 1; i++) {\n            for (int j = 1; j < NX - 1; j++) {\n                temp[i][j] = temp_new[i][j];\n            }\n        }\n    }\n}\n\nint main() {\n    clock_t start, end;\n    double cpu_time_used;\n\n    initialize();\n\n    start = clock();\n    update();\n    end = clock();\n\n    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;\n    printf("Serial Execution Time: %f seconds\\n", cpu_time_used);\n\n    return 0;\n}\n'
    write_code('heat_serial.c', serial_code_1)
    if compile_c('heat_serial.c', 'heat_serial'):
        print('\nRunning Serial Executable...')
        serial_output_1 = run_executable_1('heat_serial')
        if serial_output_1:
            print(serial_output_1)
            serial_time_match_1 = re.search('Serial Execution Time:\\s+([0-9.]+)\\s+seconds', serial_output_1)
            if serial_time_match_1:
                serial_time_1 = float(serial_time_match_1.group(1))
                print(f'Serial Execution Time: {serial_time_1:.6f} seconds\n')
            else:
                print('Failed to extract serial execution time.\n')
        else:
            print('Serial executable did not run successfully.\n')
    else:
        print('Serial compilation failed.\n')
    openmp_mpi_code_1 = '\n#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>\n#include <mpi.h>\n#include <omp.h>\n\n#define NX 3000    // Number of grid points in X-direction\n#define NY 3000    // Number of grid points in Y-direction\n#define NSTEPS 500 // Number of time steps\n\ndouble **temp;\ndouble **temp_new;\n\n// Function to allocate a 2D array dynamically\ndouble** allocate_2D(int rows, int cols) {\n    double **array = (double**) malloc(rows * sizeof(double*));\n    for(int i = 0; i < rows; i++) {\n        array[i] = (double*) malloc(cols * sizeof(double));\n    }\n    return array;\n}\n\n// Function to free a 2D array\nvoid free_2D(double **array, int rows) {\n    for(int i = 0; i < rows; i++) {\n        free(array[i]);\n    }\n    free(array);\n}\n\n// Initialize the temperature grid\nvoid initialize(int rank, int size, int local_NY, int start_row) {\n    #pragma omp parallel for collapse(2)\n    for(int i = 1; i <= local_NY; i++) { // 1 to local_NY inclusive\n        for(int j = 0; j < NX; j++) {\n            temp[i][j] = 0.0; // Initial temperature\n            // Set boundary conditions\n            if ((start_row + i -1 == 0) || (start_row + i -1 == NY -1) || j == 0 || j == NX -1) {\n                temp[i][j] = 100.0; // Hot edges\n            }\n        }\n    }\n}\n\n// Exchange ghost rows with neighboring MPI processes\nvoid exchange_ghost_rows(int rank, int size, int local_NY, MPI_Comm comm) {\n    MPI_Request requests[4];\n    int req_count = 0;\n\n    // Send to upper neighbor, receive from lower neighbor\n    if(rank != 0){\n        MPI_Isend(temp[1], NX, MPI_DOUBLE, rank -1, 0, comm, &requests[req_count++]);\n        MPI_Irecv(temp[0], NX, MPI_DOUBLE, rank -1, 1, comm, &requests[req_count++]);\n    }\n\n    // Send to lower neighbor, receive from upper neighbor\n    if(rank != size -1){\n        MPI_Isend(temp[local_NY], NX, MPI_DOUBLE, rank +1, 1, comm, &requests[req_count++]);\n        MPI_Irecv(temp[local_NY +1], NX, MPI_DOUBLE, rank +1, 0, comm, &requests[req_count++]);\n    }\n\n    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);\n}\n\n// Update the temperature grid\nvoid update(int local_NY) {\n    double alpha = 0.01; // Thermal diffusivity\n    double dx = 1.0;\n    double dy = 1.0;\n\n    #pragma omp parallel for collapse(2)\n    for(int i =1; i <= local_NY; i++) {\n        for(int j =1; j < NX -1; j++) {\n            temp_new[i][j] = temp[i][j] + alpha * (\n                (temp[i+1][j] - 2 * temp[i][j] + temp[i-1][j]) / (dx * dx) +\n                (temp[i][j+1] - 2 * temp[i][j] + temp[i][j-1]) / (dy * dy)\n            );\n        }\n    }\n\n    // Swap temp and temp_new\n    #pragma omp parallel for collapse(2)\n    for(int i =1; i <= local_NY; i++) {\n        for(int j =1; j < NX -1; j++) {\n            temp[i][j] = temp_new[i][j];\n        }\n    }\n}\n\nint main(int argc, char *argv[]) {\n    int rank, size;\n    double start_time, end_time;\n    int local_NY, start_row;\n    MPI_Init(&argc, &argv);\n    MPI_Comm comm = MPI_COMM_WORLD;\n    MPI_Comm_rank(comm, &rank);\n    MPI_Comm_size(comm, &size);\n\n    // Determine the number of rows per process\n    local_NY = NY / size;\n    int remainder = NY % size;\n    start_row = rank * local_NY + (rank < remainder ? rank : remainder);\n    local_NY += (rank < remainder) ? 1 : 0;\n\n    // Allocate local arrays with ghost rows\n    temp = allocate_2D(local_NY +2, NX);      // +2 for ghost rows\n    temp_new = allocate_2D(local_NY +2, NX);\n\n    // Initialize local grid\n    initialize(rank, size, local_NY, start_row);\n\n    MPI_Barrier(comm); // Synchronize before starting the timer\n    if(rank ==0){\n        start_time = MPI_Wtime();\n    }\n\n    // Simulation loop\n    for(int step =0; step < NSTEPS; step++) {\n        exchange_ghost_rows(rank, size, local_NY, comm);\n        update(local_NY);\n    }\n\n    MPI_Barrier(comm); // Synchronize before stopping the timer\n    if(rank ==0){\n        end_time = MPI_Wtime();\n        printf("OpenMP + MPI Execution Time: %f seconds\\n", end_time - start_time);\n    }\n\n    // Free allocated memory\n    free_2D(temp, local_NY +2);\n    free_2D(temp_new, local_NY +2);\n\n    MPI_Finalize();\n    return 0;\n}\n'
    write_code('heat_openmp_mpi.c', openmp_mpi_code_1)
    if compile_mpi_openmp('heat_openmp_mpi.c', 'heat_openmp_mpi', flags=['-fopenmp']):
        print('\nRunning OpenMP + MPI Executable with adjusted MPI processes and OpenMP threads...')
        os.environ['OMP_NUM_THREADS'] = '2'
        cpu_count = subprocess.check_output('nproc', shell=True).decode().strip()
        cpu_count = int(cpu_count)
        print(f'Number of available CPU cores: {cpu_count}')
        if cpu_count >= 4:
            mpi_processes = 2
            threads_per_process = 2
        elif cpu_count >= 2:
            mpi_processes = 2
            threads_per_process = 1
        else:
            mpi_processes = 1
            threads_per_process = 2
        print(f'Setting number of MPI processes to {mpi_processes} with {threads_per_process} OpenMP threads each.')
        os.environ['OMP_NUM_THREADS'] = str(threads_per_process)
        try:
            openmp_mpi_command = ['mpirun', '--oversubscribe', '-np', str(mpi_processes), './heat_openmp_mpi']
            openmp_mpi_output = subprocess.check_output(openmp_mpi_command, stderr=subprocess.STDOUT, text=True)
            print(openmp_mpi_output)
            openmp_mpi_time_match = re.search('OpenMP \\+ MPI Execution Time:\\s+([0-9.]+)\\s+seconds', openmp_mpi_output)
            if openmp_mpi_time_match:
                openmp_mpi_time = float(openmp_mpi_time_match.group(1))
                print(f'OpenMP + MPI Execution Time: {openmp_mpi_time:.6f} seconds\n')
            else:
                print('Failed to extract OpenMP + MPI execution time.\n')
        except subprocess.CalledProcessError as e:
            print(f'Error running OpenMP + MPI executable:\n{e.output}\n')
    else:
        print('OpenMP + MPI compilation failed.\n')
    openmp_mpi_cuda_code = '\n#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>\n#include <mpi.h>\n#include <omp.h>\n#include <cuda.h>\n\n// CUDA kernel for updating temperature\n__global__ void update_kernel(double *temp, double *temp_new, int NX, int NY, double alpha, double dx, double dy) {\n    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip boundary\n    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 to skip boundary\n\n    if(i < NY -1 && j < NX -1){\n        int idx = i * NX + j;\n        temp_new[idx] = temp[idx] + alpha * (\n            (temp[idx + NX] - 2 * temp[idx] + temp[idx - NX]) / (dx * dx) +\n            (temp[idx +1] - 2 * temp[idx] + temp[idx -1]) / (dy * dy)\n        );\n    }\n}\n\n// Function to allocate 1D arrays on the host\ndouble* allocate_1D(int size){\n    double *array;\n    cudaMallocHost(&array, size * sizeof(double));\n    return array;\n}\n\n// Function to free 1D arrays on the host\nvoid free_1D(double *array){\n    cudaFreeHost(array);\n}\n\nint main(int argc, char *argv[]) {\n    int rank, size;\n    double start_time, end_time;\n    int local_NY, start_row;\n    MPI_Init(&argc, &argv);\n    MPI_Comm comm = MPI_COMM_WORLD;\n    MPI_Comm_rank(comm, &rank);\n    MPI_Comm_size(comm, &size);\n\n    // Determine the number of rows per process\n    local_NY = NY / size;\n    int remainder = NY % size;\n    start_row = rank * local_NY + (rank < remainder ? rank : remainder);\n    local_NY += (rank < remainder) ? 1 : 0;\n\n    // Allocate host memory with ghost rows\n    int total_rows = local_NY + 2; // +2 for ghost rows\n    double *h_temp = allocate_1D(total_rows * NX);\n    double *h_temp_new = allocate_1D(total_rows * NX);\n\n    // Initialize the grid\n    for(int i =1; i <= local_NY; i++) {\n        for(int j =0; j < NX; j++) {\n            h_temp[i * NX + j] = 0.0; // Initial temperature\n            // Set boundary conditions\n            if((start_row + i -1 == 0) || (start_row + i -1 == NY -1) || j ==0 || j == NX -1){\n                h_temp[i * NX + j] = 100.0; // Hot edges\n            }\n        }\n    }\n\n    // Allocate device memory\n    double *d_temp, *d_temp_new;\n    size_t size_bytes = total_rows * NX * sizeof(double);\n    cudaMalloc((void**)&d_temp, size_bytes);\n    cudaMalloc((void**)&d_temp_new, size_bytes);\n\n    // Copy initial data to device\n    cudaMemcpy(d_temp, h_temp, size_bytes, cudaMemcpyHostToDevice);\n\n    // Define CUDA grid and block dimensions\n    dim3 blockDim(16,16);\n    dim3 gridDim( (NX + blockDim.x -1)/blockDim.x, (local_NY + blockDim.y -1)/blockDim.y );\n\n    double alpha =0.01;\n    double dx =1.0, dy =1.0;\n\n    MPI_Barrier(comm); // Synchronize before starting the timer\n    if(rank ==0){\n        start_time = MPI_Wtime();\n    }\n\n    // Simulation loop\n    for(int step =0; step < NSTEPS; step++) {\n        // Exchange ghost rows with neighboring MPI processes\n        MPI_Request requests[4];\n        int req_count =0;\n\n        // Send to upper neighbor, receive from lower neighbor\n        if(rank !=0){\n            MPI_Isend(&h_temp[1 * NX], NX, MPI_DOUBLE, rank -1, 0, comm, &requests[req_count++]);\n            MPI_Irecv(&h_temp[0 * NX], NX, MPI_DOUBLE, rank -1, 1, comm, &requests[req_count++]);\n        }\n\n        // Send to lower neighbor, receive from upper neighbor\n        if(rank != size -1){\n            MPI_Isend(&h_temp[local_NY * NX], NX, MPI_DOUBLE, rank +1, 1, comm, &requests[req_count++]);\n            MPI_Irecv(&h_temp[(local_NY +1) * NX], NX, MPI_DOUBLE, rank +1, 0, comm, &requests[req_count++]);\n        }\n\n        // Wait for all non-blocking operations to complete\n        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);\n\n        // Copy updated ghost rows to device\n        cudaMemcpy(d_temp, h_temp, size_bytes, cudaMemcpyHostToDevice);\n\n        // Launch CUDA kernel to update temperature\n        update_kernel<<<gridDim, blockDim>>>(d_temp, d_temp_new, NX, NY, alpha, dx, dy);\n        cudaDeviceSynchronize();\n\n        // Copy updated data back to host\n        cudaMemcpy(h_temp_new, d_temp_new, size_bytes, cudaMemcpyDeviceToHost);\n\n        // Swap h_temp and h_temp_new pointers\n        double *temp_ptr = h_temp;\n        h_temp = h_temp_new;\n        h_temp_new = temp_ptr;\n    }\n\n    MPI_Barrier(comm); // Synchronize before stopping the timer\n    if(rank ==0){\n        end_time = MPI_Wtime();\n        printf("OpenMP + MPI + CUDA Execution Time: %f seconds\\n", end_time - start_time);\n    }\n\n    // Free device and host memory\n    cudaFree(d_temp);\n    cudaFree(d_temp_new);\n    free_1D(h_temp);\n    free_1D(h_temp_new);\n\n    MPI_Finalize();\n    return 0;\n}\n'
    write_code('heat_openmp_mpi_cuda.cu', openmp_mpi_cuda_code)
    print('\n**Note:** Compiling the OpenMP + MPI + CUDA version in Google Colab is not recommended due to environment constraints.\n')
    print('Attempting to compile OpenMP + MPI + CUDA C code (This may fail in Colab)...')
    if compile_mpi_openmp_cuda('heat_openmp_mpi_cuda.cu', 'heat_openmp_mpi_cuda', flags=['-Xcompiler', '-fopenmp']):
        print('Compiled heat_openmp_mpi_cuda.cu successfully.')
        print('**Execution of OpenMP + MPI + CUDA executable is skipped in Colab.**\n')
    else:
        print('Compilation failed for OpenMP + MPI + CUDA executable.\n')
    print('--- Execution Time Comparison ---')
    if 'serial_time' in locals():
        print(f'Serial Execution Time: {serial_time_1:.6f} seconds')
    else:
        print('Serial Execution Time: Not Available')
    if 'openmp_mpi_time' in locals():
        print(f'OpenMP + MPI Execution Time: {openmp_mpi_time:.6f} seconds')
    else:
        print('OpenMP + MPI Execution Time: Not Available')
    if 'serial_time' in locals() and 'openmp_mpi_time' in locals() and (openmp_mpi_time > 0):
        speedup_1 = serial_time_1 / openmp_mpi_time
        total_threads = mpi_processes * threads_per_process
        efficiency_1 = speedup_1 / total_threads * 100
        print(f'Speedup: {speedup_1:.2f}x')
        print(f'Efficiency: {efficiency_1:.2f}%')
    else:
        print('Insufficient data to calculate Speedup and Efficiency.')
    print('\n--- Recommendations and Observations ---')
    if 'serial_time' in locals() and 'openmp_mpi_time' in locals():
        if openmp_mpi_time < serial_time_1:
            print('**Observation:** The OpenMP + MPI version is faster than the serial version.')
            print('**Performance Benefits Achieved Through OpenMP + MPI Parallelization.**')
        else:
            print('**Observation:** The OpenMP + MPI version is slower than the serial version.')
            print('**Possible Reasons:**')
            print('- Overhead from thread creation and synchronization.')
            print('- Limited number of physical CPU cores in the Colab environment.')
            print('- Inefficient parallelization or cache contention.')
            print('- The problem size may not be large enough to benefit from parallelization.')
            print('- OpenMP directives might not be optimally placed.\n')
            print('**Recommendations:**')
            print('- Increase the problem size (e.g., larger grid or more time steps) to better utilize parallelism.')
            print('- Experiment with different numbers of threads to find the optimal count.')
            print('- Optimize OpenMP directives, such as using appropriate scheduling strategies.')
            print('- Profile the code to identify and address bottlenecks.')
    else:
        print('Insufficient data to provide observations and recommendations.')
    return


app._unparsable_cell(
    r"""
    # ### **2D Heat Conduction Simulation: Serial, OpenMP + MPI, and OpenMP + MPI + CUDA Versions**

    # This script performs the following:
    # 1. Installs MPI and verifies CUDA installation.
    # 2. Writes and compiles the **Serial** C program with a grid size of 3000x3000.
    # 3. Writes and compiles the **OpenMP + MPI** C program with a grid size of 3000x3000.
    # 4. Writes and compiles the **OpenMP + MPI + CUDA** C program with a grid size of 3000x3000.
    # 5. Executes the Serial and OpenMP + MPI programs.
    # 6. Captures and displays their execution times.
    # 7. Attempts to compile the OpenMP + MPI + CUDA program (execution is skipped due to Colab limitations).


    # Function to write C/CUDA code to a file
    def write_code(filename, code):
        with open(filename, \"w\") as f:
            f.write(code)
        print(f\"Code written to {filename}\")

    # Function to compile C code
    def compile_c(source, output, flags=[]):
        compile_command = [\"gcc\"] + flags + [source, \"-o\", output]
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f\"Error compiling {source}:\n{result.stderr}\")
            return False
        else:
            print(f\"Compiled {source} successfully.\")
            return True

    # Function to compile MPI + OpenMP C code
    def compile_mpi_openmp(source, output, flags=[]):
        compile_command = [\"mpicc\"] + flags + [source, \"-o\", output]
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f\"Error compiling {source}:\n{result.stderr}\")
            return False
        else:
            print(f\"Compiled {source} successfully.\")
            return True

    # Function to compile MPI + OpenMP + CUDA C code
    def compile_mpi_openmp_cuda(source, output, flags=[]):
        # Attempt to locate mpi.h
        mpi_h_locations = [
            \"/usr/include/openmpi/mpi.h\",
            \"/usr/include/x86_64-linux-gnu/mpich/mpi.h\",
            \"/usr/include/mpi.h\"
        ]
        mpi_include = None
        for path in mpi_h_locations:
            if os.path.exists(path):
                mpi_include = os.path.dirname(path)
                print(f\"Found mpi.h at {path}\")
                break
        if not mpi_include:
            print(\"mpi.h not found. Cannot compile CUDA program with MPI.\")
            return False

        # Locate MPI libraries
        mpi_lib_paths = [
            \"/usr/lib/x86_64-linux-gnu\",
            \"/usr/lib\",
            \"/usr/local/lib\"
        ]
        mpi_lib = None
        for lib_path in mpi_lib_paths:
            if os.path.exists(lib_path):
                mpi_lib = lib_path
                break
        if not mpi_lib:
            print(\"MPI library not found. Cannot compile CUDA program with MPI.\")
            return False

        # Compile using nvcc with appropriate include and library paths
        compile_command = [
            \"nvcc\",
            \"-Xcompiler\", \"-fopenmp\",  # Pass OpenMP flag to host compiler
            source,
            \"-o\",
            output,
            f\"-I{mpi_include}\",          # Include path for MPI headers
            f\"-L{mpi_lib}\",              # Library path for MPI
            \"-lmpi\"                       # Link against MPI library
        ] + flags
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f\"Error compiling {source}:\n{result.stderr}\")
            return False
        else:
            print(f\"Compiled {source} successfully.\")
            return True

    # Function to run an executable and capture output
    def run_executable(exec_path):
        try:
            result = subprocess.run([f\"./{exec_path}\"], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f\"Error running {exec_path}:\n{e.output}\")
            return None

    # ------------------------------
    # 1. Install MPI and Verify CUDA
    # ------------------------------
    print(\"Installing MPI...\")
    # Update package lists and install MPI
    !apt-get update -y
    !apt-get install -y mpi-default-bin mpi-default-dev

    print(\"\nVerifying MPI installation:\")
    # Check MPI compiler version
    !mpicc --version

    print(\"\nVerifying CUDA installation:\")
    # Check CUDA compiler version
    !nvcc --version

    # ------------------------------
    # 2. Serial C Implementation
    # ------------------------------
    serial_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>

    #define NX 3000    // Number of grid points in X-direction
    #define NY 3000    // Number of grid points in Y-direction
    #define NSTEPS 500 // Number of time steps

    double temp[NY][NX];
    double temp_new[NY][NX];

    void initialize() {
        for (int i = 0; i < NY; i++) {
            for (int j = 0; j < NX; j++) {
                temp[i][j] = 0.0; // Initial temperature
                // Set boundary conditions
                if (i == 0 || i == NY - 1 || j == 0 || j == NX - 1) {
                    temp[i][j] = 100.0; // Hot edges
                }
            }
        }
    }

    void update() {
        double alpha = 0.01; // Thermal diffusivity
        double dx = 1.0;
        double dy = 1.0;

        for (int step = 0; step < NSTEPS; step++) {
            for (int i = 1; i < NY - 1; i++) {
                for (int j = 1; j < NX - 1; j++) {
                    temp_new[i][j] = temp[i][j] + alpha * (
                        (temp[i+1][j] - 2 * temp[i][j] + temp[i-1][j]) / (dx * dx) +
                        (temp[i][j+1] - 2 * temp[i][j] + temp[i][j-1]) / (dy * dy)
                    );
                }
            }
            // Copy temp_new to temp for the next iteration
            for (int i = 1; i < NY - 1; i++) {
                for (int j = 1; j < NX - 1; j++) {
                    temp[i][j] = temp_new[i][j];
                }
            }
        }
    }

    int main() {
        clock_t start, end;
        double cpu_time_used;

        initialize();

        start = clock();
        update();
        end = clock();

        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf(\"Serial Execution Time: %f seconds\\n\", cpu_time_used);

        return 0;
    }
    \"\"\"

    # Write Serial C Code
    write_code(\"heat_serial.c\", serial_code)

    # Compile Serial C Code
    if compile_c(\"heat_serial.c\", \"heat_serial\"):
        # Run Serial Executable
        print(\"\nRunning Serial Executable...\")
        serial_output = run_executable(\"heat_serial\")
        if serial_output:
            print(serial_output)
            # Extract Serial Execution Time
            serial_time_match = re.search(r\"Serial Execution Time:\s+([0-9.]+)\s+seconds\", serial_output)
            if serial_time_match:
                serial_time = float(serial_time_match.group(1))
                print(f\"Serial Execution Time: {serial_time:.6f} seconds\n\")
            else:
                print(\"Failed to extract serial execution time.\n\")
        else:
            print(\"Serial executable did not run successfully.\n\")
    else:
        print(\"Serial compilation failed.\n\")

    # ------------------------------
    # 3. OpenMP + MPI C Implementation
    # ------------------------------
    openmp_mpi_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    #include <mpi.h>
    #include <omp.h>

    #define NX 3000    // Number of grid points in X-direction
    #define NY 3000    // Number of grid points in Y-direction
    #define NSTEPS 500 // Number of time steps

    double **temp;
    double **temp_new;

    // Function to allocate a 2D array dynamically
    double** allocate_2D(int rows, int cols) {
        double **array = (double**) malloc(rows * sizeof(double*));
        for(int i = 0; i < rows; i++) {
            array[i] = (double*) malloc(cols * sizeof(double));
        }
        return array;
    }

    // Function to free a 2D array
    void free_2D(double **array, int rows) {
        for(int i = 0; i < rows; i++) {
            free(array[i]);
        }
        free(array);
    }

    // Initialize the temperature grid
    void initialize(int rank, int size, int local_NY, int start_row) {
        #pragma omp parallel for collapse(2)
        for(int i = 1; i <= local_NY; i++) { // 1 to local_NY inclusive
            for(int j = 0; j < NX; j++) {
                temp[i][j] = 0.0; // Initial temperature
                // Set boundary conditions
                if ((start_row + i -1 == 0) || (start_row + i -1 == NY -1) || j == 0 || j == NX -1) {
                    temp[i][j] = 100.0; // Hot edges
                }
            }
        }
    }

    // Exchange ghost rows with neighboring MPI processes
    void exchange_ghost_rows(int rank, int size, int local_NY, MPI_Comm comm) {
        MPI_Request requests[4];
        int req_count = 0;

        // Send to upper neighbor, receive from lower neighbor
        if(rank != 0){
            MPI_Isend(temp[1], NX, MPI_DOUBLE, rank -1, 0, comm, &requests[req_count++]);
            MPI_Irecv(temp[0], NX, MPI_DOUBLE, rank -1, 1, comm, &requests[req_count++]);
        }

        // Send to lower neighbor, receive from upper neighbor
        if(rank != size -1){
            MPI_Isend(temp[local_NY], NX, MPI_DOUBLE, rank +1, 1, comm, &requests[req_count++]);
            MPI_Irecv(temp[local_NY +1], NX, MPI_DOUBLE, rank +1, 0, comm, &requests[req_count++]);
        }

        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }

    // Update the temperature grid
    void update(int local_NY) {
        double alpha = 0.01; // Thermal diffusivity
        double dx = 1.0;
        double dy = 1.0;

        #pragma omp parallel for collapse(2)
        for(int i =1; i <= local_NY; i++) {
            for(int j =1; j < NX -1; j++) {
                temp_new[i][j] = temp[i][j] + alpha * (
                    (temp[i+1][j] - 2 * temp[i][j] + temp[i-1][j]) / (dx * dx) +
                    (temp[i][j+1] - 2 * temp[i][j] + temp[i][j-1]) / (dy * dy)
                );
            }
        }

        // Swap temp and temp_new
        #pragma omp parallel for collapse(2)
        for(int i =1; i <= local_NY; i++) {
            for(int j =1; j < NX -1; j++) {
                temp[i][j] = temp_new[i][j];
            }
        }
    }

    int main(int argc, char *argv[]) {
        int rank, size;
        double start_time, end_time;
        int local_NY, start_row;
        MPI_Init(&argc, &argv);
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // Determine the number of rows per process
        local_NY = NY / size;
        int remainder = NY % size;
        start_row = rank * local_NY + (rank < remainder ? rank : remainder);
        local_NY += (rank < remainder) ? 1 : 0;

        // Allocate local arrays with ghost rows
        temp = allocate_2D(local_NY +2, NX);      // +2 for ghost rows
        temp_new = allocate_2D(local_NY +2, NX);

        // Initialize local grid
        initialize(rank, size, local_NY, start_row);

        MPI_Barrier(comm); // Synchronize before starting the timer
        if(rank ==0){
            start_time = MPI_Wtime();
        }

        // Simulation loop
        for(int step =0; step < NSTEPS; step++) {
            exchange_ghost_rows(rank, size, local_NY, comm);
            update(local_NY);
        }

        MPI_Barrier(comm); // Synchronize before stopping the timer
        if(rank ==0){
            end_time = MPI_Wtime();
            printf(\"OpenMP + MPI Execution Time: %f seconds\\n\", end_time - start_time);
        }

        // Free allocated memory
        free_2D(temp, local_NY +2);
        free_2D(temp_new, local_NY +2);

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Write OpenMP + MPI C Code
    write_code(\"heat_openmp_mpi.c\", openmp_mpi_code)

    # Compile OpenMP + MPI C Code
    if compile_mpi_openmp(\"heat_openmp_mpi.c\", \"heat_openmp_mpi\", flags=[\"-fopenmp\"]):
        # Run OpenMP + MPI Executable with appropriate MPI processes and OpenMP threads
        print(\"\nRunning OpenMP + MPI Executable with adjusted MPI processes and OpenMP threads...\")

        # Set OpenMP threads
        os.environ[\"OMP_NUM_THREADS\"] = \"2\"

        # Determine available CPU cores
        cpu_count = subprocess.check_output(\"nproc\", shell=True).decode().strip()
        cpu_count = int(cpu_count)
        print(f\"Number of available CPU cores: {cpu_count}\")

        # Set number of MPI processes and OpenMP threads based on available cores to prevent oversubscription
        if cpu_count >= 4:
            mpi_processes = 2
            threads_per_process = 2
        elif cpu_count >= 2:
            mpi_processes = 2
            threads_per_process = 1
        else:
            mpi_processes = 1
            threads_per_process = 2  # All threads in one process

        print(f\"Setting number of MPI processes to {mpi_processes} with {threads_per_process} OpenMP threads each.\")

        # Update OMP_NUM_THREADS accordingly
        os.environ[\"OMP_NUM_THREADS\"] = str(threads_per_process)

        # Execute the OpenMP + MPI program
        try:
            # Using subprocess to capture the output
            # Use --oversubscribe to allow running more MPI processes than available slots if necessary
            openmp_mpi_command = [\"mpirun\", \"--oversubscribe\", \"-np\", str(mpi_processes), \"./heat_openmp_mpi\"]
            openmp_mpi_output = subprocess.check_output(openmp_mpi_command, stderr=subprocess.STDOUT, text=True)
            print(openmp_mpi_output)
            # Extract OpenMP + MPI Execution Time
            openmp_mpi_time_match = re.search(r\"OpenMP \+ MPI Execution Time:\s+([0-9.]+)\s+seconds\", openmp_mpi_output)
            if openmp_mpi_time_match:
                openmp_mpi_time = float(openmp_mpi_time_match.group(1))
                print(f\"OpenMP + MPI Execution Time: {openmp_mpi_time:.6f} seconds\n\")
            else:
                print(\"Failed to extract OpenMP + MPI execution time.\n\")
        except subprocess.CalledProcessError as e:
            print(f\"Error running OpenMP + MPI executable:\n{e.output}\n\")
    else:
        print(\"OpenMP + MPI compilation failed.\n\")

    # ------------------------------
    # 4. OpenMP + MPI + CUDA C Implementation
    # ------------------------------
    openmp_mpi_cuda_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    #include <mpi.h>
    #include <omp.h>
    #include <cuda.h>

    // Define the macros
    #define NX 3000    // Number of grid points in X-direction
    #define NY 3000    // Number of grid points in Y-direction
    #define NSTEPS 500 // Number of time steps

    // CUDA kernel for updating temperature
    __global__ void update_kernel(double *temp, double *temp_new, int NX, int NY, double alpha, double dx, double dy) {
        int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip boundary
        int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 to skip boundary

        if(i < NY -1 && j < NX -1){
            int idx = i * NX + j;
            temp_new[idx] = temp[idx] + alpha * (
                (temp[idx + NX] - 2 * temp[idx] + temp[idx - NX]) / (dx * dx) +
                (temp[idx +1] - 2 * temp[idx] + temp[idx -1]) / (dy * dy)
            );
        }
    }

    // Function to allocate 1D arrays on the host
    double* allocate_1D(int size){
        double *array;
        cudaMallocHost(&array, size * sizeof(double));
        return array;
    }

    // Function to free 1D arrays on the host
    void free_1D(double *array){
        cudaFreeHost(array);
    }

    int main(int argc, char *argv[]) {
        int rank, size;
        double start_time, end_time;
        int local_NY, start_row;
        MPI_Init(&argc, &argv);
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // Determine the number of rows per process
        local_NY = NY / size;
        int remainder = NY % size;
        start_row = rank * local_NY + (rank < remainder ? rank : remainder);
        local_NY += (rank < remainder) ? 1 : 0;

        // Allocate host memory with ghost rows
        int total_rows = local_NY + 2; // +2 for ghost rows
        double *h_temp = allocate_1D(total_rows * NX);
        double *h_temp_new = allocate_1D(total_rows * NX);

        // Initialize the grid
        for(int i =1; i <= local_NY; i++) {
            for(int j =0; j < NX; j++) {
                h_temp[i * NX + j] = 0.0; // Initial temperature
                // Set boundary conditions
                if((start_row + i -1 == 0) || (start_row + i -1 == NY -1) || j ==0 || j == NX -1){
                    h_temp[i * NX + j] = 100.0; // Hot edges
                }
            }
        }

        // Allocate device memory
        double *d_temp, *d_temp_new;
        size_t size_bytes = total_rows * NX * sizeof(double);
        cudaMalloc((void**)&d_temp, size_bytes);
        cudaMalloc((void**)&d_temp_new, size_bytes);

        // Copy initial data to device
        cudaMemcpy(d_temp, h_temp, size_bytes, cudaMemcpyHostToDevice);

        // Define CUDA grid and block dimensions
        dim3 blockDim(16,16);
        dim3 gridDim( (NX + blockDim.x -1)/blockDim.x, (local_NY + blockDim.y -1)/blockDim.y );

        double alpha =0.01;
        double dx =1.0, dy =1.0;

        MPI_Barrier(comm); // Synchronize before starting the timer
        if(rank ==0){
            start_time = MPI_Wtime();
        }

        // Simulation loop
        for(int step =0; step < NSTEPS; step++) {
            // Exchange ghost rows with neighboring MPI processes
            MPI_Request requests[4];
            int req_count =0;

            // Send to upper neighbor, receive from lower neighbor
            if(rank !=0){
                MPI_Isend(&h_temp[1 * NX], NX, MPI_DOUBLE, rank -1, 0, comm, &requests[req_count++]);
                MPI_Irecv(&h_temp[0 * NX], NX, MPI_DOUBLE, rank -1, 1, comm, &requests[req_count++]);
            }

            // Send to lower neighbor, receive from upper neighbor
            if(rank != size -1){
                MPI_Isend(&h_temp[local_NY * NX], NX, MPI_DOUBLE, rank +1, 1, comm, &requests[req_count++]);
                MPI_Irecv(&h_temp[(local_NY +1) * NX], NX, MPI_DOUBLE, rank +1, 0, comm, &requests[req_count++]);
            }

            // Wait for all non-blocking operations to complete
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

            // Copy updated ghost rows to device
            cudaMemcpy(d_temp, h_temp, size_bytes, cudaMemcpyHostToDevice);

            // Launch CUDA kernel to update temperature
            update_kernel<<<gridDim, blockDim>>>(d_temp, d_temp_new, NX, NY, alpha, dx, dy);
            cudaDeviceSynchronize();

            // Copy updated data back to host
            cudaMemcpy(h_temp_new, d_temp_new, size_bytes, cudaMemcpyDeviceToHost);

            // Swap h_temp and h_temp_new pointers
            double *temp_ptr = h_temp;
            h_temp = h_temp_new;
            h_temp_new = temp_ptr;
        }

        MPI_Barrier(comm); // Synchronize before stopping the timer
        if(rank ==0){
            end_time = MPI_Wtime();
            printf(\"OpenMP + MPI + CUDA Execution Time: %f seconds\\n\", end_time - start_time);
        }

        // Free device and host memory
        cudaFree(d_temp);
        cudaFree(d_temp_new);
        free_1D(h_temp);
        free_1D(h_temp_new);

        MPI_Finalize();
        return 0;
    }
    \"\"\"

    # Write OpenMP + MPI + CUDA C Code
    write_code(\"heat_openmp_mpi_cuda.cu\", openmp_mpi_cuda_code)

    # Compile OpenMP + MPI + CUDA C Code
    print(\"\n**Note:** Compiling the OpenMP + MPI + CUDA version in Google Colab is not recommended due to environment constraints.\n\")
    print(\"Attempting to compile OpenMP + MPI + CUDA C code (This may fail in Colab)...\")

    if compile_mpi_openmp_cuda(\"heat_openmp_mpi_cuda.cu\", \"heat_openmp_mpi_cuda\", flags=[\"-Xcompiler\", \"-fopenmp\"]):
        print(\"Compiled heat_openmp_mpi_cuda.cu successfully.\")
        # **Execution is skipped due to Colab limitations**
        print(\"**Execution of OpenMP + MPI + CUDA executable is skipped in Colab.**\n\")
    else:
        print(\"Compilation failed for OpenMP + MPI + CUDA executable.\n\")

    # ------------------------------
    # 5. Performance Comparison
    # ------------------------------
    print(\"--- Execution Time Comparison ---\")
    if 'serial_time' in locals():
        print(f\"Serial Execution Time: {serial_time:.6f} seconds\")
    else:
        print(\"Serial Execution Time: Not Available\")

    if 'openmp_mpi_time' in locals():
        print(f\"OpenMP + MPI Execution Time: {openmp_mpi_time:.6f} seconds\")
    else:
        print(\"OpenMP + MPI Execution Time: Not Available\")

    # Calculate Speedup and Efficiency
    if 'serial_time' in locals() and 'openmp_mpi_time' in locals() and openmp_mpi_time > 0:
        speedup = serial_time / openmp_mpi_time
        total_threads = mpi_processes * threads_per_process
        efficiency = (speedup / total_threads) * 100
        print(f\"Speedup: {speedup:.2f}x\")
        print(f\"Efficiency: {efficiency:.2f}%\")
    else:
        print(\"Insufficient data to calculate Speedup and Efficiency.\")

    # ------------------------------
    # 6. Recommendations and Observations
    # ------------------------------
    print(\"\n--- Recommendations and Observations ---\")
    if 'serial_time' in locals() and 'openmp_mpi_time' in locals():
        if openmp_mpi_time < serial_time:
            print(\"**Observation:** The OpenMP + MPI version is faster than the serial version.\")
            print(\"**Performance Benefits Achieved Through OpenMP + MPI Parallelization.**\")
        else:
            print(\"**Observation:** The OpenMP + MPI version is slower than the serial version.\")
            print(\"**Possible Reasons:**\")
            print(\"- Overhead from thread creation and synchronization.\")
            print(\"- Limited number of physical CPU cores in the Colab environment.\")
            print(\"- Inefficient parallelization or cache contention.\")
            print(\"- The problem size may not be large enough to benefit from parallelization.\")
            print(\"- OpenMP directives might not be optimally placed.\n\")
            print(\"**Recommendations:**\")
            print(\"- Increase the problem size (e.g., larger grid or more time steps) to better utilize parallelism.\")
            print(\"- Experiment with different numbers of threads to find the optimal count.\")
            print(\"- Optimize OpenMP directives, such as using appropriate scheduling strategies.\")
            print(\"- Profile the code to identify and address bottlenecks.\")
    else:
        print(\"Insufficient data to provide observations and recommendations.\")
    """,
    name="_"
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

