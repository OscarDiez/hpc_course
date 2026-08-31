import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to GPU Computing

        In today's computational landscape, GPUs (Graphics Processing Units) have evolved beyond their original purpose of rendering graphics. They have become essential tools for high-performance computing (HPC), capable of performing a vast number of calculations simultaneously. This ability to handle massive parallelism makes GPUs particularly useful for scientific simulations, machine learning, and other computationally intensive tasks.

        In this section, we'll explore the basics of GPU computing, including the architecture of GPUs and the programming models that allow us to harness their power. Specifically, we'll focus on two main approaches: CUDA and OpenACC.

        ## What is GPU Computing?

        GPU computing involves using a GPU to perform computations that would otherwise be handled by a CPU. Unlike CPUs, which have a few cores optimized for sequential processing, GPUs have thousands of smaller, more efficient cores designed for parallel tasks. This makes GPUs ideal for workloads that can be broken down into many smaller, independent tasks.

        ## Why Use GPUs in HPC?

        - **Massive Parallelism**: GPUs can execute thousands of threads simultaneously, making them well-suited for parallel algorithms.
        - **High Throughput**: With their large number of cores, GPUs can process a significant amount of data in parallel, leading to faster execution times for many applications.
        - **Energy Efficiency**: GPUs can often perform more computations per watt compared to CPUs, making them a more energy-efficient choice for large-scale computations.

        In the following sections, we will dive into the basics of CUDA and OpenACC, two popular programming models for GPU computing.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## CUDA Basics

        CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows developers to use NVIDIA GPUs for general-purpose processing (an approach known as GPGPU, General-Purpose computing on Graphics Processing Units).

        ### Key Concepts in CUDA

        - **Thread**: The smallest unit of processing. Each thread runs a copy of the kernel (a function that runs on the GPU).
        - **Block**: A group of threads that can cooperate amongst themselves via shared memory. Blocks are organized into a grid.
        - **Grid**: A collection of blocks. The grid represents the entire execution space for a given kernel.

        Let's start by writing a simple CUDA program to perform matrix multiplication, a common operation in many scientific and engineering applications.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## CUDA Basics

        CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows developers to use NVIDIA GPUs for general-purpose processing (an approach known as GPGPU, General-Purpose computing on Graphics Processing Units).

        ### Key Concepts in CUDA

        - **Thread**: The smallest unit of processing. Each thread runs a copy of the kernel (a function that runs on the GPU).
        - **Block**: A group of threads that can cooperate amongst themselves via shared memory. Blocks are organized into a grid.
        - **Grid**: A collection of blocks. The grid represents the entire execution space for a given kernel.

        Let's start by writing a simple CUDA program to perform matrix multiplication, a common operation in many scientific and engineering applications.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Check the GPU available in Google Colab
    !nvidia-smi
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to CUDA Programming

        In this example, you will learn how to write a basic CUDA program that adds two vectors. We will define the vector addition on the GPU, allocate memory on the GPU, and then copy the result back to the CPU.

        The steps are as follows:
        1. Define the vectors on the CPU.
        2. Copy the vectors from the CPU to the GPU.
        3. Perform the vector addition on the GPU.
        4. Copy the result back to the CPU.
        5. Print the result.

        Let's write, compile, and run a simple CUDA program for vector addition.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 1: Writing the CUDA C code to a file for vector addition
    cuda_code = \"\"\"
    #include <stdio.h>
    #include <cuda_runtime.h>

    // CUDA kernel for vector addition
    __global__ void vectorAdd(float *A, float *B, float *C, int n) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < n) {
            C[i] = A[i] + B[i];
        }
    }

    int main() {
        int n = 1024;  // Vector size

        // Allocate memory on the host (CPU)
        size_t size = n * sizeof(float);
        float *h_A = (float *)malloc(size);
        float *h_B = (float *)malloc(size);
        float *h_C = (float *)malloc(size);

        // Initialize vectors A and B
        for (int i = 0; i < n; ++i) {
            h_A[i] = i * 1.0f;
            h_B[i] = i * 2.0f;
        }

        // Allocate memory on the device (GPU)
        float *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, size);
        cudaMalloc((void **)&d_B, size);
        cudaMalloc((void **)&d_C, size);

        // Copy vectors from host to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Launch the vector addition kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

        // Copy the result vector from device to host
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        // Print the result (only showing the first 10 elements)
        printf(\"Result of vector addition (first 10 elements):\\n\");
        for (int i = 0; i < 10; ++i) {
            printf(\"%f + %f = %f\\n\", h_A[i], h_B[i], h_C[i]);
        }

        // Free memory on device and host
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);

        return 0;
    }
    \"\"\"

    # Step 2: Save the CUDA code to a file
    with open('vector_addition.cu', 'w') as f:
        f.write(cuda_code)

    # Step 3: Compile the CUDA program using nvcc
    !nvcc -o vector_addition vector_addition.cu

    # Step 4: Run the CUDA program
    !./vector_addition
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to Matrix Multiplication using CUDA

        In this section, we will explore how to perform matrix multiplication using CUDA. Matrix multiplication is a fundamental operation in many scientific and engineering applications. GPUs are highly suited for this task because they allow us to perform many operations in parallel, making the multiplication of large matrices significantly faster compared to running on the CPU.

        ### Key Concepts:
        1. **Thread Hierarchy**: CUDA organizes computation into a grid of blocks, where each block contains multiple threads. Each thread operates on a small part of the data, and together they process the entire matrix.
        2. **Global Memory**: Data is stored in global memory on the GPU, which can be accessed by all threads. In this example, matrices `A`, `B`, and `C` are stored in global memory.
        3. **Kernel**: A CUDA kernel is a function that runs on the GPU, with each thread executing an instance of the kernel. In this example, the kernel `matrixMul` computes one element of the output matrix `C`.

        ### Matrix Multiplication Overview:
        Matrix multiplication involves computing the dot product of rows of matrix `A` with columns of matrix `B` to produce an element in matrix `C`. Each thread calculates one element of the output matrix `C`.

        In this example, we:
        1. **Define a kernel (`matrixMul`)** that performs the matrix multiplication.
        2. **Allocate memory** on both the CPU (host) and the GPU (device).
        3. **Copy data** from the CPU to the GPU.
        4. **Launch the kernel** on the GPU, where each thread computes one element of the result matrix.
        5. **Copy the result** back to the CPU and display part of the result matrix.

        This example uses matrices of size 512x512, but CUDA allows us to scale this to much larger matrices efficiently.

        ### Task:
        Run the CUDA code for matrix multiplication using 512x512 matrices and observe the result printed on the console (a part of the resulting matrix). This will help you understand how to perform parallel matrix computations on the GPU.

        """
    )
    return


@app.cell
def _():
    # Write the CUDA C code for matrix multiplication to a file
    cuda_code = """
    #include <stdio.h>
    #include <cuda_runtime.h>

    #define N 512  // Define the size of the matrix

    __global__ void matrixMul(float *A, float *B, float *C, int n) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0;

        if (row < n && col < n) {
            for (int i = 0; i < n; ++i) {
                sum += A[row * n + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }

    int main() {
        int size = N * N * sizeof(float);

        // Allocate memory on the host
        float *h_A = (float *)malloc(size);
        float *h_B = (float *)malloc(size);
        float *h_C = (float *)malloc(size);

        // Initialize matrices A and B with more variation
        for (int i = 0; i < N * N; ++i) {
            int row = i / N;
            int col = i % N;
            h_A[i] = (row + 1) + (col % 5);  // Create variation across rows and columns
            h_B[i] = (col + 1) * (row % 3 + 1);  // Create variation across rows and columns
        }

        // Allocate memory on the device
        float *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, size);
        cudaMalloc((void **)&d_B, size);
        cudaMalloc((void **)&d_C, size);

        // Copy matrices from host to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Define the block size and grid size
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch the matrix multiplication kernel
        matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        // Copy the result matrix back to the host
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        // Print the result matrix
        printf("Result matrix C (only showing a part of it):\\n");
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 10; ++j) {
                printf("%f ", h_C[i * N + j]);
            }
            printf("\\n");
        }

        // Free memory on device and host
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);

        return 0;
    }
    """

    with open('matrix_mul.cu', 'w') as f:
        f.write(cuda_code)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding the CUDA Matrix Multiplication Code

        This CUDA program performs matrix multiplication, a fundamental operation in many scientific and engineering applications, by leveraging the parallel processing power of GPUs.

        - **Matrix Size Definition**: `#define N 512` sets the matrix dimensions to 512x512, which can be adjusted as needed.

        - **CUDA Kernel Function (`matrixMul`)**: The kernel function performs the actual multiplication. Each thread computes a single element of the result matrix `C` by iterating over a row of matrix `A` and a column of matrix `B`. The `blockIdx` and `threadIdx` are used to determine the position of the element each thread is responsible for.

        - **Memory Management**:
          - **Host Memory**: Memory for matrices `A`, `B`, and `C` is allocated on the host (CPU) using `malloc`.
          - **Device Memory**: Corresponding memory on the GPU is allocated using `cudaMalloc`.
          - **Data Transfer**: The matrices `A` and `B` are copied from the host to the GPU before computation, and the result matrix `C` is copied back from the GPU to the host after the kernel execution.

        - **Grid and Block Configuration**: The grid and block dimensions are set using `dim3`, with a 16x16 block size to distribute the workload among the GPU threads efficiently.

        - **Kernel Launch**: The `matrixMul` kernel is launched with the specified grid and block dimensions to perform the matrix multiplication on the GPU.

        - **Result Output**: After computation, a portion of the result matrix `C` is printed to verify the output.

        - **Memory Cleanup**: Finally, both host and device memory are freed to prevent memory leaks.

        This example highlights the essential steps in a CUDA program, including memory management, kernel configuration, and parallel computation, providing a foundation for more complex GPU-accelerated applications.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Running the CUDA Program

        Next, we'll compile and run the CUDA program to see how GPU-based matrix multiplication works. The result matrix should reflect the combined computation of the two input matrices.

        We'll use `nvcc`, the CUDA compiler, to compile the code and then execute the compiled binary. If everything is set up correctly, you should see the output of the matrix multiplication displayed below.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Compile the CUDA program
    !nvcc -o matrix_mul matrix_mul.cu
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Run the CUDA program
    !./matrix_mul
    """,
    name="_"
)


@app.cell
def _():
    cuda_code_1 = '\n#include <stdio.h>\n#include <cuda_runtime.h>\n\n#define N 512  // Define the size of the matrix\n\n// Kernel for dense matrix multiplication\n__global__ void denseMatrixMul(float *A, float *B, float *C, int n) {\n    int row = blockIdx.y * blockDim.y + threadIdx.y;\n    int col = blockIdx.x * blockDim.x + threadIdx.x;\n    float sum = 0.0;\n\n    if (row < n && col < n) {\n        for (int i = 0; i < n; ++i) {\n            sum += A[row * n + i] * B[i * n + col];\n        }\n        C[row * n + col] = sum;\n    }\n}\n\n// Kernel for sparse matrix multiplication (simplified for demo purposes)\n__global__ void sparseMatrixMul(int *rowPtr, int *colInd, float *values, float *B, float *C, int n, int nnz) {\n    int row = blockIdx.y * blockDim.y + threadIdx.y;\n    int col = blockIdx.x * blockDim.x + threadIdx.x;\n    float sum = 0.0;\n\n    if (row < n && col < n) {\n        for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; ++idx) {\n            int j = colInd[idx];\n            sum += values[idx] * B[j * n + col];\n        }\n        C[row * n + col] = sum;\n    }\n}\n\nint main() {\n    int size = N * N * sizeof(float);\n\n    // Allocate memory on the host for dense matrices\n    float *h_A = (float *)malloc(size);\n    float *h_B = (float *)malloc(size);\n    float *h_C_dense = (float *)malloc(size);\n    float *h_C_sparse = (float *)malloc(size);\n\n\n    // Initialize dense matrices A and B with varying values\n    for (int i = 0; i < N * N; ++i) {\n        h_A[i] = ((i % N) + 1) * ((i / N) % 3 + 1);  // Example: values are products of row and column indices with some variation\n        h_B[i] = ((i / N) + 1) * 0.5f * ((i % N) % 4 + 1);  // Example: values depend on both row and column indices with multiplication factors\n    }\n\n\n    // Allocate memory on the device for dense matrices\n    float *d_A, *d_B, *d_C_dense;\n    cudaMalloc((void **)&d_A, size);\n    cudaMalloc((void **)&d_B, size);\n    cudaMalloc((void **)&d_C_dense, size);\n\n    // Copy matrices from host to device for dense multiplication\n    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);\n    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);\n\n    // Define the block size and grid size for dense multiplication\n    dim3 threadsPerBlock(16, 16);\n    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,\n                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);\n\n    // Launch the dense matrix multiplication kernel\n    denseMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_dense, N);\n\n    // Copy the result matrix back to the host for dense multiplication\n    cudaMemcpy(h_C_dense, d_C_dense, size, cudaMemcpyDeviceToHost);\n\n    // Print results for dense matrix multiplication\n    printf("Result matrix C (Dense - showing a part of it):\\n");\n    for (int i = 0; i < 10; ++i) {\n        for (int j = 0; j < 10; ++j) {\n            printf("%f ", h_C_dense[i * N + j]);\n        }\n        printf("\\n");\n    }\n\n    // Example of sparse matrix multiplication\n    // Note: Sparse matrix in CSR format is represented by rowPtr, colInd, and values arrays\n    int nnz = 3 * N;  // Example: 3 non-zero entries per row\n    int *h_rowPtr = (int *)malloc((N + 1) * sizeof(int));\n    int *h_colInd = (int *)malloc(nnz * sizeof(int));\n    float *h_values = (float *)malloc(nnz * sizeof(float));\n\n\n    // Initialize sparse matrix A in CSR format\n    for (int i = 0; i < N; ++i) {\n        h_rowPtr[i] = i * 3;  // 3 non-zeros per row\n        for (int j = 0; j < 3; ++j) {\n            h_colInd[i * 3 + j] = (i + j) % N;  // Example column indices\n            h_values[i * 3 + j] = 1.0f;  // Non-zero values\n        }\n    }\n    h_rowPtr[N] = nnz;  // Last entry of rowPtr\n\n    // Allocate memory on the device for sparse matrix multiplication\n    int *d_rowPtr, *d_colInd;\n    float *d_values, *d_C_sparse;\n    cudaMalloc((void **)&d_rowPtr, (N + 1) * sizeof(int));\n    cudaMalloc((void **)&d_colInd, nnz * sizeof(int));\n    cudaMalloc((void **)&d_values, nnz * sizeof(float));\n    cudaMalloc((void **)&d_C_sparse, size);\n\n    // Copy sparse matrix data from host to device\n    cudaMemcpy(d_rowPtr, h_rowPtr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);\n    cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice);\n    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);\n    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);  // Reuse B from dense multiplication\n\n    // Launch the sparse matrix multiplication kernel\n    sparseMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_rowPtr, d_colInd, d_values, d_B, d_C_sparse, N, nnz);\n\n    // Copy the result matrix back to the host for sparse multiplication\n    cudaMemcpy(h_C_sparse, d_C_sparse, size, cudaMemcpyDeviceToHost);\n\n    // Print results for sparse matrix multiplication\n    printf("\\nResult matrix C (Sparse - showing a part of it):\\n");\n    for (int i = 0; i < 10; ++i) {\n        for (int j = 0; j < 10; ++j) {\n            printf("%f ", h_C_sparse[i * N + j]);\n        }\n        printf("\\n");\n    }\n\n    // Free memory on device and host\n    cudaFree(d_A);\n    cudaFree(d_B);\n    cudaFree(d_C_dense);\n    cudaFree(d_rowPtr);\n    cudaFree(d_colInd);\n    cudaFree(d_values);\n    cudaFree(d_C_sparse);\n    free(h_A);\n    free(h_B);\n    free(h_C_dense);\n    free(h_C_sparse);\n    free(h_rowPtr);\n    free(h_colInd);\n    free(h_values);\n\n    return 0;\n}\n'
    with open('matrix_mul.cu', 'w') as f_1:
        f_1.write(cuda_code_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding the CUDA Code for Dense and Sparse Matrix Multiplication

        This CUDA program demonstrates two different approaches to matrix multiplication: dense and sparse. Matrix multiplication is a core operation in many high-performance computing applications, and understanding both dense and sparse implementations is crucial for optimizing performance depending on the data characteristics.

        ### Dense Matrix Multiplication
        - **Kernel Function (`denseMatrixMul`)**: This kernel computes the product of two dense matrices `A` and `B`, storing the result in matrix `C`. Each thread is responsible for calculating a single element in the result matrix `C` by iterating over a row in `A` and a column in `B`.
        - **Host and Device Memory Management**:
          - **Host Memory**: Allocated for matrices `A`, `B`, and `C_dense` using `malloc`.
          - **Device Memory**: Corresponding memory is allocated on the GPU using `cudaMalloc`.
          - **Data Transfer**: Matrices `A` and `B` are copied from host to device memory before the kernel is executed, and the resulting matrix `C_dense` is copied back to the host after the kernel execution.
        - **Kernel Launch Configuration**: The matrix multiplication kernel is launched with a grid and block configuration determined by the size of the matrix and the number of threads per block, maximizing GPU resource utilization.

        ### Sparse Matrix Multiplication
        - **Sparse Matrix Representation**: Sparse matrices store only non-zero elements to save memory and computation time. Here, the sparse matrix `A` is represented in Compressed Sparse Row (CSR) format using three arrays: `rowPtr`, `colInd`, and `values`.
          - **`rowPtr`**: Indicates the start and end of each row in the `colInd` and `values` arrays.
          - **`colInd`**: Stores the column indices of the non-zero elements.
          - **`values`**: Stores the actual non-zero values of the matrix.
        - **Kernel Function (`sparseMatrixMul`)**: This kernel performs matrix multiplication using the sparse matrix `A` and dense matrix `B`, storing the result in matrix `C_sparse`. Each thread computes a single element in `C_sparse` by accessing only the non-zero elements in `A`.
        - **Host and Device Memory Management**:
          - **Host Memory**: Additional memory is allocated for the sparse matrix representation (`rowPtr`, `colInd`, and `values`).
          - **Device Memory**: Memory for the sparse matrix components and the result matrix `C_sparse` is allocated on the GPU.
          - **Data Transfer**: Sparse matrix data is copied from host to device before kernel execution, and the resulting matrix `C_sparse` is copied back to the host afterward.

        ### Result Verification and Output
        - After executing both the dense and sparse matrix multiplication kernels, the program prints a portion of the resulting matrices `C_dense` and `C_sparse`. This output allows for verification of the correctness of the matrix multiplication operations.

        ### Memory Cleanup
        - The program ensures proper cleanup by freeing both host and device memory after the computation, preventing memory leaks and ensuring efficient use of resources.

        This code provides a practical example of how to implement and compare dense and sparse matrix multiplication on a GPU using CUDA, demonstrating the advantages of using sparse matrices when dealing with data that contains many zeroes.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Compile the CUDA program
    !nvcc -lcusparse -o matrix_mul matrix_mul.cu
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Run the CUDA program
    !./matrix_mul
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Experiment with Dense and Sparse Matrix Multiplication in CUDA

        In this exercise, you will modify the existing CUDA program to better understand the difference between dense and sparse matrix multiplication. The program currently performs both dense and sparse matrix multiplication on a 512x512 matrix using CUDA.

        ### Task 1: Modify the Sparse Matrix Structure
        The current sparse matrix is initialized with **3 non-zero entries per row**. Modify the program so that:
        1. The number of non-zero entries per row is **random** (between 1 and 5 non-zero elements per row).
        2. Observe how the output changes when the matrix sparsity varies.

        ### Task 2: Change the Matrix Size
        The matrix size is currently defined as **512x512**. Modify the matrix size to:
        1. **Increase the matrix size** to 1024x1024.
        2. **Compare the performance** of dense and sparse matrix multiplication for different matrix sizes (e.g., 512x512 vs. 1024x1024).

        ### Hints:
        - You can modify the number of non-zero entries per row by using a random function (e.g., `rand()`).
        - The matrix size can be modified by changing the value of the constant `N`.
        - Check the output and runtime for both dense and sparse matrices as you change these values.

        After making these changes, run the program and observe the results. This will help you understand the impact of sparsity on matrix multiplication performance and how CUDA handles larger matrices.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OpenACC
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to OpenACC

        While CUDA is a powerful and flexible tool for GPU programming, it requires detailed management of the GPU's memory and parallelization. OpenACC is a higher-level programming model designed to simplify the process of parallelizing code. With OpenACC, you can accelerate your existing C, C++, and Fortran applications without requiring in-depth knowledge of GPU architecture.

        ### Key Concepts in OpenACC

        - **Directives**: OpenACC uses compiler directives (pragmas) to specify which parts of the code should run on the GPU. This allows for incremental parallelization of existing code.
        - **Parallel Region**: A block of code that is executed by multiple threads in parallel on the GPU.
        - **Data Region**: Specifies the movement of data between the CPU and GPU memory.

        Let's take a look at a similar matrix multiplication example using OpenACC.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Install wget and the necessary dependencies
    !apt-get install -y wget build-essential

    # Download the NVIDIA HPC SDK
    !wget https://developer.download.nvidia.com/hpc-sdk/23.1/nvhpc_2023_231_Linux_x86_64_cuda_12.0.tar.gz -O nvhpc.tar.gz

    # Extract the downloaded tarball
    !tar -xzf nvhpc.tar.gz
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Navigate to the extracted folder and run the install script non-interactively
    !cd nvhpc_2023_231_Linux_x86_64_cuda_12.0 && yes \"\" | ./install --installpath /usr/local/nvhpc --accept --silent
    """,
    name="_"
)


@app.cell
def _():
    import os
    # Set up environment variables for the NVIDIA HPC SDK installed in /opt/nvidia/hpc_sdk
    os.environ['PATH'] = '/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin:' + os.environ['PATH']
    os.environ['LD_LIBRARY_PATH'] = '/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/lib:' + os.environ['LD_LIBRARY_PATH']
    return


app._unparsable_cell(
    r"""
    # Check if nvc is in the PATH
    !which nvc
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to OpenACC: Simple Vector Addition

        OpenACC is a directive-based programming model that simplifies parallel programming for multi-core CPUs and GPUs. With just a few directives, OpenACC allows us to offload computation to GPUs, without needing to manage the low-level details of GPU programming (like CUDA).

        ### Key Concepts of OpenACC:
        1. **Directives**: OpenACC uses `#pragma acc` directives to indicate which parts of the code should be parallelized.
        2. **Parallelization**: We can parallelize loops easily using OpenACC, allowing multiple threads to work on different iterations of the loop simultaneously.
        3. **Data Transfer**: OpenACC handles data movement between the CPU (host) and the GPU (device) with `copy`, `copyin`, `copyout`, and `create`.

        ### Simple Vector Addition:
        In this example, we will perform a simple vector addition. The program will:
        1. **Initialize two vectors** `A` and `B` on the CPU (host).
        2. **Use OpenACC directives** to parallelize the addition of these vectors on the GPU (device).
        3. **Store the result** in vector `C` and print the first few values of the result.

        ### Task:
        Run the provided OpenACC code for vector addition using vectors of size 1024. Observe the result printed on the console. This will help you understand how OpenACC simplifies parallel programming.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 1: Writing the OpenACC C code to a file for simple vector addition
    openacc_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>

    #define N 1024  // Define the size of the vectors

    int main() {
        // Allocate memory on the host
        float *A = (float *)malloc(N * sizeof(float));
        float *B = (float *)malloc(N * sizeof(float));
        float *C = (float *)malloc(N * sizeof(float));

        // Initialize vectors A and B with some values
        for (int i = 0; i < N; i++) {
            A[i] = i * 1.0f;
            B[i] = i * 2.0f;
        }

        // Perform vector addition using OpenACC
        #pragma acc data copyin(A[0:N], B[0:N]), copyout(C[0:N])
        {
            #pragma acc parallel loop
            for (int i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
            }
        }

        // Print a portion of the result vector
        printf(\"Result vector C (first 10 elements):\\n\");
        for (int i = 0; i < 10; i++) {
            printf(\"%f + %f = %f\\n\", A[i], B[i], C[i]);
        }

        // Free memory
        free(A);
        free(B);
        free(C);

        return 0;
    }
    \"\"\"

    # Step 2: Save the OpenACC code to a file
    with open('vector_add_openacc.c', 'w') as f:
        f.write(openacc_code)

    # Step 3: Compile the OpenACC code using nvc (NVIDIA C Compiler) from the HPC SDK
    !nvc -acc -o vector_add_openacc vector_add_openacc.c -Minfo=accel

    # Step 4: Run the OpenACC program
    !./vector_add_openacc
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to OpenACC: Data Management and Atomics

        In this section, we will explore how to use **data management** and **atomic operations** in OpenACC. These are crucial for efficiently managing memory transfers between the CPU and GPU, and for ensuring that multiple threads can safely update shared data.

        ### Key Concepts of OpenACC:
        1. **Data Directives**: OpenACC uses data directives to control the movement of data between the CPU (host) and the GPU (device). Some important clauses include:
           - **`copy`**: Allocates memory on the GPU, copies the data from the CPU to the GPU, and copies it back to the CPU after the computation is done.
           - **`copyin`**: Copies data from the CPU to the GPU, but does not copy it back.
           - **`copyout`**: Allocates memory on the GPU and copies the result back to the CPU after computation.
        2. **Atomic Operations**: Atomic operations ensure that updates to shared data by multiple threads are performed one at a time, preventing race conditions. In OpenACC, you can use `#pragma acc atomic` to enforce this.

        ### Example: Safe Parallel Increment with Atomic Operations
        In this example, we will:
        1. **Initialize an array** and a shared counter.
        2. **Use OpenACC to parallelize the loop** that increments the shared counter based on array values.
        3. Use **atomic operations** to ensure that the counter is safely updated by multiple threads.

        ### Task:
        Run the provided OpenACC code, which uses `copyin`, `copyout`, and `copy` clauses for data management and `#pragma acc atomic` for safe updates to the shared counter. Observe how atomic operations work to avoid race conditions.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 1: Writing the OpenACC C code to a file for data management and atomic operations
    openacc_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>

    #define N 1000  // Define the size of the array

    int main() {
        // Allocate memory on the host
        int *A = (int *)malloc(N * sizeof(int));
        int sum = 0;  // Shared variable for atomic increment

        // Initialize the array A
        for (int i = 0; i < N; i++) {
            A[i] = 1;  // All values are set to 1 for easy summing
        }

        // Perform parallel summing using OpenACC with atomic operation
        #pragma acc data copyin(A[0:N]), copy(sum)
        {
            #pragma acc parallel loop
            for (int i = 0; i < N; i++) {
                #pragma acc atomic
                sum += A[i];  // Atomic operation to safely increment shared variable
            }
        }

        // Print the result
        printf(\"The total sum of the array is: %d\\n\", sum);

        // Free memory
        free(A);

        return 0;
    }
    \"\"\"

    # Step 2: Save the OpenACC code to a file
    with open('data_mgmt_atomic_openacc.c', 'w') as f:
        f.write(openacc_code)

    # Step 3: Compile the OpenACC code using nvc (NVIDIA C Compiler) from the HPC SDK
    !nvc -acc -o data_mgmt_atomic_openacc data_mgmt_atomic_openacc.c -Minfo=accel

    # Step 4: Run the OpenACC program
    !./data_mgmt_atomic_openacc
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Experiment with Data Management and Atomic Operations in OpenACC

        In this exercise, you will modify the existing OpenACC program to better understand how atomic operations and data management work.

        ### Task 1: Remove Atomic and Observe the Impact
        The current program uses **`#pragma acc atomic`** to ensure that the variable `sum` is incremented safely when multiple threads are updating it. Modify the code by:
        1. **Removing the `#pragma acc atomic` directive**.
        2. Run the program and observe what happens when atomic operations are removed. Do you get the same sum? What issues arise?

        ### Task 2: Parallelize Initialization of the Array
        Currently, the array `A` is initialized on the CPU in a serial manner. Modify the program so that:
        1. The initialization of array `A` is done in parallel using OpenACC.
        2. Use **`#pragma acc parallel loop`** to parallelize the loop that initializes `A`.

        ### Hints:
        - Removing the atomic directive may cause race conditions. Observe how the output changes and consider why the result is different.
        - Use **data management clauses** like `copyin` and `copyout` to manage the data transfer between the host (CPU) and device (GPU) when initializing the array in parallel.

        After making these changes, run the program and observe how removing atomic operations and parallelizing the array initialization impact the result and performance.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OpenACC: Understanding `parallel` vs `kernels`

        OpenACC provides two main directives to parallelize code: **`parallel`** and **`kernels`**. Both are used to execute code on the GPU, but they serve slightly different purposes.

        ### Key Differences:
        1. **`parallel` Directive**:
           - The `parallel` directive gives you **explicit control** over which parts of the code are parallelized.
           - You need to manually specify the loops or blocks of code that should be run in parallel.
           - Suitable when you know exactly how you want to parallelize your loops.

        2. **`kernels` Directive**:
           - The `kernels` directive lets the compiler automatically decide how to parallelize the code.
           - The compiler analyzes the code and identifies parallel loops.
           - Suitable for more **general parallelization** where you trust the compiler to do the work.

        ### Example: Vector Addition with `parallel` and `kernels`
        In this example, we will:
        1. **Use the `parallel` directive** to parallelize the vector addition manually.
        2. **Use the `kernels` directive** to let the compiler decide the parallelization.

        Both implementations perform the same operation, but the choice of directive determines how much control you have over the parallelization process.

        ### Task:
        Run the provided OpenACC code to see how vector addition is performed using both `parallel` and `kernels`. Compare the outputs and observe how each directive behaves.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 1: Writing the OpenACC C code to a file for parallel vs kernels
    openacc_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>

    #define N 1024  // Define the size of the vectors

    int main() {
        // Allocate memory on the host
        float *A = (float *)malloc(N * sizeof(float));
        float *B = (float *)malloc(N * sizeof(float));
        float *C_parallel = (float *)malloc(N * sizeof(float));
        float *C_kernels = (float *)malloc(N * sizeof(float));

        // Initialize vectors A and B with some values
        for (int i = 0; i < N; i++) {
            A[i] = i * 1.0f;
            B[i] = i * 2.0f;
        }

        // Vector addition using OpenACC with the 'parallel' directive
        #pragma acc data copyin(A[0:N], B[0:N]), copyout(C_parallel[0:N])
        {
            #pragma acc parallel loop
            for (int i = 0; i < N; i++) {
                C_parallel[i] = A[i] + B[i];
            }
        }

        // Vector addition using OpenACC with the 'kernels' directive
        #pragma acc data copyin(A[0:N], B[0:N]), copyout(C_kernels[0:N])
        {
            #pragma acc kernels
            for (int i = 0; i < N; i++) {
                C_kernels[i] = A[i] + B[i];
            }
        }

        // Print a portion of the result vectors to compare
        printf(\"Result of vector addition using 'parallel' (first 10 elements):\\n\");
        for (int i = 0; i < 10; i++) {
            printf(\"%f + %f = %f\\n\", A[i], B[i], C_parallel[i]);
        }

        printf(\"\\nResult of vector addition using 'kernels' (first 10 elements):\\n\");
        for (int i = 0; i < 10; i++) {
            printf(\"%f + %f = %f\\n\", A[i], B[i], C_kernels[i]);
        }

        // Free memory
        free(A);
        free(B);
        free(C_parallel);
        free(C_kernels);

        return 0;
    }
    \"\"\"

    # Step 2: Save the OpenACC code to a file
    with open('parallel_vs_kernels_openacc.c', 'w') as f:
        f.write(openacc_code)

    # Step 3: Compile the OpenACC code using nvc (NVIDIA C Compiler) from the HPC SDK
    !nvc -acc -o parallel_vs_kernels_openacc parallel_vs_kernels_openacc.c -Minfo=accel

    # Step 4: Run the OpenACC program
    !./parallel_vs_kernels_openacc
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to Matrix Multiplication using OpenACC

        In this section, we will explore how to perform matrix multiplication using OpenACC, a programming model designed for parallel computing on multi-core CPUs and GPUs. OpenACC provides simple directives for accelerating code without needing to manage low-level details like memory transfers between the CPU and GPU.

        ### Key Concepts of OpenACC:
        1. **Directives**: OpenACC uses compiler directives (e.g., `#pragma acc`) to specify which parts of the code should be parallelized and run on the GPU.
        2. **Data Management**: OpenACC allows developers to manage data transfer between the CPU (host) and GPU (device) using `copyin`, `copyout`, and `copy` clauses.
        3. **Parallel Loops**: OpenACC can parallelize loops automatically using `#pragma acc parallel loop`.

        ### Matrix Multiplication Overview:
        In this example, we will perform matrix multiplication with OpenACC using a 512x512 matrix. The steps are as follows:
        1. Allocate and initialize the matrices `A` and `B`.
        2. Use OpenACC directives to perform matrix multiplication on the GPU.
        3. Copy the result back to the CPU and print a part of the resulting matrix `C`.

        ### Task:
        Run the provided OpenACC code for matrix multiplication using a 512x512 matrix and observe the result printed on the console (a part of the resulting matrix). This will help you understand how OpenACC simplifies parallel programming.

        """
    )
    return


@app.cell
def _():
    openacc_code = '\n#include <stdio.h>\n#include <stdlib.h>\n\n#define N 512  // Define the size of the matrix\n\nint main() {\n    int size = N * N * sizeof(float);\n\n    // Allocate memory on the host\n    float *A = (float *)malloc(size);\n    float *B = (float *)malloc(size);\n    float *C = (float *)malloc(size);\n\n    // Initialize matrices A and B with varying values\n    for (int i = 0; i < N * N; i++) {\n        int row = i / N;\n        int col = i % N;\n        A[i] = (float)((row + 1) * (col + 2)) * 0.5f;  // Initialize A with values based on row and column\n        B[i] = (float)((col + 1) * (row + 2)) * 0.3f;  // Initialize B with values based on row and column\n    }\n\n    // Perform matrix multiplication using OpenACC\n    #pragma acc data copyin(A[0:N*N], B[0:N*N]), copyout(C[0:N*N])\n    {\n        #pragma acc parallel loop collapse(2)\n        for (int i = 0; i < N; i++) {\n            for (int j = 0; j < N; j++) {\n                float sum = 0.0f;\n                for (int k = 0; k < N; k++) {\n                    sum += A[i * N + k] * B[k * N + j];\n                }\n                C[i * N + j] = sum;\n            }\n        }\n    }\n\n    // Print a portion of the result matrix\n    printf("Result matrix C (only showing a part of it):\\n");\n    for (int i = 0; i < 10; i++) {\n        for (int j = 0; j < 10; j++) {\n            printf("%f ", C[i * N + j]);\n        }\n        printf("\\n");\n    }\n\n    // Free memory\n    free(A);\n    free(B);\n    free(C);\n\n    return 0;\n}\n'
    with open('matrix_mul_openacc.c', 'w') as f_2:
        f_2.write(openacc_code)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Experiment with OpenACC Matrix Multiplication

        In this exercise, you will modify the existing OpenACC matrix multiplication code to better understand how OpenACC handles parallelization and performance.

        ### Task 1: Modify the Block Size
        The current OpenACC code uses a **`collapse(2)`** clause to parallelize the outer two loops of matrix multiplication. Modify the code so that:
        1. **Remove the `collapse(2)` clause** and parallelize just the outer loop (i.e., only the row loop).
        2. Run the program and compare the performance and results. Does removing the `collapse(2)` affect the computation or speed?

        ### Task 2: Test with Different Matrix Sizes
        The matrix size is currently defined as **512x512**. Modify the matrix size to:
        1. **Increase the matrix size to 1024x1024**.
        2. Run the program and observe the difference in runtime with a larger matrix. How does the change in matrix size affect the performance?

        ### Hints:
        - You can modify the matrix size by changing the value of the constant **`N`**.
        - Observe the difference between parallelizing the outer loop and collapsing two loops. This will help you understand how loop parallelization affects performance.

        After making these changes, run the program and observe how the modifications impact both the runtime and the output. This exercise will help you learn how OpenACC parallelizes loops and how matrix size influences performance.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding the OpenACC Code for Matrix Multiplication

        This OpenACC program demonstrates matrix multiplication, a fundamental operation in many scientific computing tasks, using GPU acceleration. OpenACC is a user-friendly directive-based programming model that allows developers to parallelize code easily and target accelerators like GPUs without deep knowledge of GPU architecture.

        ### Matrix Initialization
        - **Matrix Size and Memory Allocation**:
          - The matrix size `N` is defined as 512, resulting in matrices `A`, `B`, and `C` of size 512x512.
          - Memory for these matrices is allocated on the host using `malloc`, with each matrix occupying `N*N*sizeof(float)` bytes.
  
        - **Initializing Matrices**:
          - Matrices `A` and `B` are initialized with varying values based on their row and column indices. This variation ensures that each element in the matrices has a unique value, allowing for meaningful computation and results.
          - Matrix `A` is filled with values calculated as `((row + 1) * (col + 2)) * 0.5f`.
          - Matrix `B` is initialized with values calculated as `((col + 1) * (row + 2)) * 0.3f`.

        ### Parallel Matrix Multiplication with OpenACC
        - **OpenACC Directives**:
          - The `#pragma acc data` directive is used to manage data transfer between the host and the device (GPU). It ensures that matrices `A` and `B` are copied to the device before the computation, and matrix `C` is copied back to the host after the computation.
          - The `#pragma acc parallel loop collapse(2)` directive specifies that the following nested loops should be parallelized, allowing multiple threads on the GPU to perform the matrix multiplication concurrently. The `collapse(2)` clause indicates that both loops should be parallelized together, enabling efficient use of the GPU's parallel architecture.

        - **Matrix Multiplication**:
          - The nested loops iterate over the rows of `A` and columns of `B` to compute each element of the result matrix `C`. Each element `C[i * N + j]` is calculated as the dot product of the `i-th` row of `A` and the `j-th` column of `B`.
          - The inner loop accumulates the product of corresponding elements from `A` and `B`, storing the sum in `C`.

        ### Result Display and Memory Cleanup
        - **Output**:
          - After computation, the program prints a portion of the result matrix `C` (the top-left 10x10 submatrix). This partial display allows for quick verification of the computation's correctness.
  
        - **Memory Management**:
          - The program concludes by freeing the allocated memory for matrices `A`, `B`, and `C` on the host, ensuring efficient use of resources and preventing memory leaks.

        This code provides a practical example of how to use OpenACC for parallelizing a common computational task (matrix multiplication) and demonstrates how OpenACC simplifies the process of leveraging GPU acceleration.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Running the OpenACC Program

        Now that we've written our OpenACC code for matrix multiplication, let's compile and run it. The OpenACC program should produce a result matrix similar to the CUDA version but with potentially simpler code.

        OpenACC abstracts much of the complexity of GPU programming, making it easier to parallelize existing C or Fortran code. Let's see how it performs.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Compile the OpenACC code using nvc (NVIDIA C Compiler) from the HPC SDK
    !nvc -acc -o matrix_mul_openacc matrix_mul_openacc.c -Minfo=accel
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Run the OpenACC program
    !./matrix_mul_openacc
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Comparing CUDA and OpenACC

        Both CUDA and OpenACC are powerful tools for leveraging the computational power of GPUs, but they serve different purposes and target different audiences:

        - **CUDA**: Offers more control over the hardware and is ideal for developers who need fine-grained optimization and are familiar with GPU architecture. It requires manual management of memory and parallelization, but this can lead to highly optimized code.

        - **OpenACC**: Provides a higher-level, more abstracted approach to GPU programming. It's easier to use for those who want to accelerate their applications without diving deeply into the specifics of GPU hardware. OpenACC is often used to incrementally parallelize existing applications.

        ### Key Takeaways

        - **Flexibility vs. Ease of Use**: CUDA is more flexible but requires more effort, while OpenACC is easier to use but might not offer the same level of optimization.
        - **Learning Curve**: CUDA has a steeper learning curve compared to OpenACC.
        - **Performance**: Depending on the application and how well the code is optimized, CUDA might offer better performance, but OpenACC can still deliver significant speedups with much less effort.

        ### Conclusion

        In this session, we explored the basics of GPU computing, focusing on two popular approaches: CUDA and OpenACC. We've implemented matrix multiplication in both frameworks and compared their usage. Understanding both CUDA and OpenACC allows you to choose the best tool for your specific needs, whether that’s maximum performance or ease of development.

        Continue experimenting with these examples and explore how you can leverage GPU computing for your projects!

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

