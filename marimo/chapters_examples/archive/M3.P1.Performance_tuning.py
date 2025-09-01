import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Performance Optimization in HPC

        ## Introduction

        In this notebook, we will explore the fundamental techniques for optimizing code performance in High-Performance Computing (HPC) environments. Performance optimization is crucial for fully exploiting the capabilities of HPC architectures. By understanding and applying these techniques, you can significantly reduce the runtime of your computational tasks, making them more efficient and scalable.

        This practice is essential in HPC as it allows for better resource utilization, reduced costs, and the ability to solve larger and more complex problems. We will cover various optimization strategies, including code profiling, memory hierarchy optimization, and the use of high-performance libraries.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Optimizing Code for HPC Architectures

        ### 2.1 Code Profiling and Analysis

        Before optimizing any code, it's essential to understand where the bottlenecks are. Profiling tools help identify the most time-consuming parts of your code, which are the primary candidates for optimization.

        ### 2.2 Loop Unrolling and Vectorization

        Loop unrolling and vectorization are common techniques used to enhance the performance of loops, which are often the most time-consuming parts of computational code.

        ### 2.3 Memory Access Patterns and Cache Utilization

        Efficient memory access patterns and effective use of the CPU cache can dramatically speed up your programs.

        """
    )
    return


@app.cell
def _():
    import cProfile
    import random

    def matrix_multiply(A, B):
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])
        if cols_A != rows_B:
            raise ValueError('Cannot multiply matrices: number of columns in A must be equal to number of rows in B.')
        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] = result[i][j] + A[i][k] * B[k][j]
        return result

    def create_random_matrix(rows, cols):
        return [[random.random() for _ in range(cols)] for _ in range(rows)]
    A = create_random_matrix(300, 300)
    B = create_random_matrix(300, 300)
    cProfile.run('matrix_multiply(A, B)')
    return (random,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        The above code uses Python's `cProfile` to profile a matrix multiplication function. Profiling helps identify the parts of the code that consume the most computational resources, allowing us to focus our optimization efforts effectively.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Memory Hierarchy and Data Locality

        ### 3.1 Understanding Memory Hierarchy

        Memory hierarchy, from registers to cache and RAM, plays a critical role in the performance of HPC applications. Optimizing for memory hierarchy can significantly reduce data access times.

        ### 3.2 Data Locality

        Data locality refers to the use of data elements within close proximity in memory, reducing cache misses and improving overall performance.

        """
    )
    return


@app.cell
def _(random):
    import time

    def create_random_matrix_1(rows, cols):
        return [[random.random() for _ in range(cols)] for _ in range(rows)]

    def row_wise_sum(matrix):
        total = []
        for row in matrix:
            total.append(sum(row))
        return total

    def column_wise_sum(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        total = [0] * cols
        for col in range(cols):
            for row in range(rows):
                total[col] = total[col] + matrix[row][col]
        return total
    matrix = create_random_matrix_1(2000, 2000)
    start_time = time.time()
    row_sums = row_wise_sum(matrix)
    end_time = time.time()
    print(f'Time taken for row-wise sum: {end_time - start_time:.2f} seconds')
    start_time = time.time()
    column_sums = column_wise_sum(matrix)
    end_time = time.time()
    print(f'Time taken for column-wise sum: {end_time - start_time:.2f} seconds')
    return (time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        In the above example, we measure the performance impact of accessing matrix elements row-wise versus column-wise. Due to the way memory is structured, row-wise access is typically faster because it accesses contiguous memory locations, which is more cache-friendly.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding Modules in HPC Environments

        In High-Performance Computing (HPC) environments, managing software dependencies can be complex due to the variety of libraries and software packages that different applications may require. The **Modules** system is designed to simplify this process. It allows users to dynamically modify their environment (e.g., `PATH`, `LD_LIBRARY_PATH`) to load or unload specific software packages or libraries on-demand, ensuring compatibility and reproducibility across different applications.

        Modules make it easier to:
        - Load specific versions of software or libraries
        - Avoid conflicts between different software versions
        - Automatically set environment variables required by certain software

        ### Main Module Commands Explained with Examples

        1. **Listing Available Modules**
           - This command shows a list of all modules currently available on the cluster that can be loaded into your environment.
           - **Command:**
             ```bash
             module avail
             ```
           - **Example Output:**
             ```bash
             -------------------- /opt/modulefiles --------------------
             openblas/0.3.17    fftw/3.3.8    gcc/9.3.0    python/3.8.10
             intel-mkl/2021.2   cuda/11.2     hdf5/1.10.6
             ```
             This output lists available versions of software and libraries like OpenBLAS, FFTW, and Python.

        ---

        2. **Loading a Module**
           - To use a specific software package, load its corresponding module. This automatically sets the necessary environment variables.
           - **Command:**
             ```bash
             module load <module-name>/<version>
             ```
           - **Example:**
             ```bash
             module load openblas/0.3.17
             ```
             This command loads OpenBLAS version 0.3.17 into your environment.

        ---

        3. **Unloading a Module**
           - To remove a module from your environment, use the `unload` command.
           - **Command:**
             ```bash
             module unload <module-name>/<version>
             ```
           - **Example:**
             ```bash
             module unload openblas/0.3.17
             ```
             This will unload OpenBLAS from your environment, removing any changes it made to your environment variables.

        ---

        4. **Listing Loaded Modules**
           - To check which modules are currently loaded in your environment, use the `list` command.
           - **Command:**
             ```bash
             module list
             ```
           - **Example Output:**
             ```bash
             Currently Loaded Modules:
             1) gcc/9.3.0   2) openblas/0.3.17   3) python/3.8.10
             ```
             This shows that GCC, OpenBLAS, and Python are currently loaded.

        ---

        5. **Switching Between Module Versions**
           - You can switch between different versions of a module using the `swap` command.
           - **Command:**
             ```bash
             module swap <old-module>/<old-version> <new-module>/<new-version>
             ```
           - **Example:**
             ```bash
             module swap openblas/0.3.17 openblas/0.3.9
             ```
             This will unload OpenBLAS version 0.3.17 and load version 0.3.9.

        ---

        6. **Getting Information About a Module**
           - The `show` command provides detailed information about a module, including the environment variables it modifies and paths to executables or libraries.
           - **Command:**
             ```bash
             module show <module-name>/<version>
             ```
           - **Example:**
             ```bash
             module show openblas/0.3.17
             ```
           - **Example Output:**
             ```bash
             -------------------------------------------------------------------
             /opt/modulefiles/openblas/0.3.17:

             module-whatis  "OpenBLAS: An optimized BLAS library"
             prepend-path    PATH /opt/openblas/0.3.17/bin
             prepend-path    LD_LIBRARY_PATH /opt/openblas/0.3.17/lib
             setenv          OPENBLAS_VERSION 0.3.17
             -------------------------------------------------------------------
             ```
             This output shows how OpenBLAS modifies your environment when loaded, such as adding directories to the `PATH` and `LD_LIBRARY_PATH`.

        ---

        7. **Searching for Modules**
           - If you're not sure about the exact name or version of a module, you can search for it using the `spider` command.
           - **Command:**
             ```bash
             module spider <module-name>
             ```
           - **Example:**
             ```bash
             module spider openblas
             ```
             This will list all available versions of OpenBLAS and show how to load them.

        ---

        8. **Purging All Loaded Modules**
           - If you want to remove all loaded modules and reset your environment to its default state, use the `purge` command.
           - **Command:**
             ```bash
             module purge
             ```
           - **Example:**
             ```bash
             module purge
             ```
             This will unload all currently loaded modules, restoring your environment to its initial state.

        ---

        9. **Saving and Restoring Module Sets**
           - You can save the current set of loaded modules to easily restore them later. This is helpful when working on multiple projects that require different sets of modules.
           - **Save the current module environment**:
             ```bash
             module save <set-name>
             ```
           - **Restore a saved module environment**:
             ```bash
             module restore <set-name>
             ```
           - **Example:**
             ```bash
             module save my-project
             module restore my-project
             ```
             The `save` command saves the current modules as a named set (`my-project`), and the `restore` command reloads that set when needed.

        ---

        ### Example: Loading BLAS and LAPACK Libraries

        In HPC, numerical libraries like **BLAS** (Basic Linear Algebra Subprograms) and **LAPACK** (Linear Algebra Package) are often used for linear algebra computations. These libraries are highly optimized and may be provided by modules such as **OpenBLAS** or **Intel MKL**.

        Here is an example of how to load the `openblas/0.3.17` module for a program that requires BLAS and LAPACK:

        ```bash
        # Load the OpenBLAS module
        module load openblas/0.3.17

        # Verify that the module has been loaded
        module list

        """
    )
    return


app._unparsable_cell(
    r"""
    !module avail
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    !module list
    """,
    name="_"
)


@app.cell
def _():
    import subprocess
    subprocess.run('# Load the OpenBLAS module\nmodule load openblas/0.3.17\n\n# Verify that the module has been loaded\nmodule list', shell=True)
    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. High-Performance Libraries for Scientific Computing

        Leveraging high-performance libraries can save development time and ensure that your code is optimized for modern HPC architectures.

        ### 4.1 Using BLAS and LAPACK for Linear Algebra

        BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra Package) are standard libraries that provide optimized implementations of basic linear algebra routines.
        ## Introduction to BLAS and LAPACK

        BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra Package) are highly optimized libraries that provide standard routines for performing common linear algebra operations. These libraries are widely used in scientific computing, engineering, and data analysis due to their efficiency and portability across different hardware architectures.

        ### Why Use BLAS and LAPACK?

        BLAS and LAPACK are particularly useful for the following reasons:
        - **Performance**: These libraries are fine-tuned to utilize the underlying hardware, making them highly efficient for operations such as matrix multiplication, solving linear systems, and eigenvalue problems.
        - **Parallelism**: BLAS and LAPACK implementations often leverage multi-threading and hardware acceleration (e.g., using vectorized instructions), making them ideal for high-performance computing (HPC) environments.
        - **Portability**: BLAS and LAPACK are available on a wide range of platforms and are included in many high-performance libraries like Intel MKL, OpenBLAS, and ATLAS.

        ### What Will We Do?

        In this exercise, we will:
        1. Perform matrix multiplication using BLAS's `dgemm` routine, which is specifically optimized for this task.
        2. Solve a system of linear equations using LAPACK's `dgesv` routine, which finds the solution to `AX = B` using LU factorization.

        By leveraging these libraries, we can efficiently handle large linear algebra problems that are common in scientific and engineering applications.


        """
    )
    return


@app.cell
def _():
    import os

    # Step 1: Write the C program to a file
    c_program = """
    #include <stdio.h>
    #include <stdlib.h>

    // Declare BLAS and LAPACK routines
    extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
                       double *alpha, double *A, int *lda, double *B, int *ldb,
                       double *beta, double *C, int *ldc);

    extern void dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv,
                       double *B, int *ldb, int *info);

    void print_matrix(const char* name, double *matrix, int rows, int cols) {
        printf("%s:\\n", name);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%f ", matrix[i * cols + j]);
            }
            printf("\\n");
        }
    }

    int main() {
        // Example: Matrix multiplication using dgemm (BLAS)
    
        // Matrices A (3x3), B (3x3), and C (3x3) for A * B = C
        double A[9] = {1.0, 2.0, 3.0, 
                       4.0, 5.0, 6.0, 
                       7.0, 8.0, 9.0};
        double B[9] = {9.0, 8.0, 7.0, 
                       6.0, 5.0, 4.0, 
                       3.0, 2.0, 1.0};
        double C[9];
    
        int m = 3, n = 3, k = 3;  // Dimensions of matrices
        double alpha = 1.0, beta = 0.0;
    
        // Matrix multiplication C = alpha * A * B + beta * C
        dgemm_("N", "N", &m, &n, &k, &alpha, A, &m, B, &n, &beta, C, &m);
    
        // Print the result of A * B
        print_matrix("Matrix C (A * B)", C, m, n);

        // Example: Solving a linear system using dgesv (LAPACK)
    
        // A (3x3) and B (3x1), solve A * X = B
        double A2[9] = {3.0, 2.0, -1.0,
                        2.0, -2.0, 4.0,
                        -1.0, 0.5, -1.0};
        double B2[3] = {1.0, -2.0, 0.0};  // Right-hand side
    
        int ipiv[3];  // Pivot indices
        int info;     // Return info
        int nrhs = 1; // Number of right-hand sides
    
        // Solve the system of equations A * X = B
        dgesv_(&m, &nrhs, A2, &m, ipiv, B2, &m, &info);
    
        if (info == 0) {
            // Print the solution
            print_matrix("Solution to A * X = B", B2, m, nrhs);
        } else {
            printf("An error occurred: dgesv returned info = %d\\n", info);
        }

        return 0;
    }
    """

    # Save the C program to a file
    c_filename = "blas_lapack_example.c"
    with open(c_filename, "w") as c_file:
        c_file.write(c_program)

    print(f"C program written to {c_filename}")
    return (os,)


@app.cell
def _(subprocess):
    # Step 3: Compile the C program using IMKL (Intel Math Kernel Library) instead of OpenBLAS
    compile_command = "gcc -o blas_lapack_example blas_lapack_example.c -lmkl_rt"

    # Run the compile command
    compile_process = subprocess.run(compile_command, shell=True, capture_output=True, text=True)

    # Output compilation results
    if compile_process.returncode == 0:
        print("Compilation successful.")
    else:
        print(f"Compilation failed:\n{compile_process.stderr}")

    # Step 4: Run the compiled binary and capture the output
    if compile_process.returncode == 0:
        run_command = "./blas_lapack_example"
        run_process = subprocess.run(run_command, shell=True, capture_output=True, text=True)

        # Output the results of the program
        if run_process.returncode == 0:
            print(f"Program Output:\n{run_process.stdout}")
        else:
            print(f"Program Error:\n{run_process.stderr}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of the Code

        The C program provided uses BLAS and LAPACK routines to perform two common linear algebra operations: matrix multiplication and solving a system of linear equations.

        ### Matrix Multiplication with BLAS (`dgemm`)

        The program first demonstrates matrix multiplication using the BLAS routine `dgemm`. This function performs the operation:

        \[
        C = \alpha \times A \times B + \beta \times C
        \]

        Where:
        - `A`, `B`, and `C` are matrices,
        - `\alpha` and `\beta` are scalar values.

        In the code:
        - We define two 3x3 matrices `A` and `B` and multiply them to produce matrix `C`.
        - The `dgemm_` function from BLAS is called with appropriate arguments, performing the matrix multiplication and storing the result in `C`.
        - The function `print_matrix` is then used to print the resulting matrix `C`.

        ### Solving a Linear System with LAPACK (`dgesv`)

        Next, the program solves the system of linear equations:

        \[
        A \times X = B
        \]

        Where `A` is a matrix, and `B` is a vector. The LAPACK routine `dgesv` is used for this purpose, which computes the solution `X` by performing LU factorization of matrix `A`.

        In the code:
        - We define a 3x3 matrix `A2` and a 3x1 vector `B2`.
        - The `dgesv_` function from LAPACK is used to solve the system. The solution vector `X` (which replaces `B2` after the call) is printed using the `print_matrix` function.
        - The `dgesv_` function internally performs LU decomposition and uses the result to compute the solution. The pivot indices required for LU factorization are stored in the array `ipiv`.

        ### Why Use BLAS and LAPACK?

        BLAS and LAPACK provide highly efficient and reliable methods to perform these operations. By using these libraries, you can benefit from:
        - **Speed**: These routines are often faster than hand-written matrix multiplication or equation solvers.
        - **Stability**: LAPACK uses numerically stable algorithms to ensure the accuracy of the solutions.
        - **Flexibility**: The same routines can handle matrices of various sizes, allowing scalability to larger problems.

        This exercise demonstrates the power of using these libraries in computational tasks, particularly for high-performance computing or large-scale data analysis.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Advantages of Using Optimized Libraries vs. Writing Your Own Code

        When performing complex linear algebra operations, you have two options: write your own code or use highly optimized libraries like BLAS and LAPACK. While it may seem tempting to implement these algorithms yourself for learning purposes, using optimized libraries offers many advantages, especially in High-Performance Computing (HPC) environments.

        ### 1. **Performance**
        Optimized libraries like BLAS and LAPACK are carefully designed and fine-tuned to take advantage of modern CPU architectures. They leverage low-level optimizations, such as vectorization, multi-threading, and cache utilization, to ensure that matrix operations are performed as quickly as possible.

        - **Custom Code**: Your implementation may work well for small matrices but will likely struggle with large datasets, leading to increased runtime and resource consumption.
        - **Optimized Libraries**: By using BLAS and LAPACK, you can achieve orders of magnitude faster performance for large matrices and complex operations, as these libraries are built to scale.

        ### 2. **Stability and Accuracy**
        Numerical stability is a key concern in scientific computing. Libraries like LAPACK use robust, tested algorithms to ensure that operations like solving linear systems are performed with maximum precision.

        - **Custom Code**: Writing your own solver may introduce numerical inaccuracies, especially for large matrices or ill-conditioned systems.
        - **Optimized Libraries**: LAPACK's routines ensure that matrix operations are numerically stable, providing accurate solutions even for complex problems.

        ### 3. **Scalability and Multi-node Execution**
        In HPC environments, scaling applications across multiple nodes is essential for handling large datasets. Optimized libraries like BLAS and LAPACK are designed to work efficiently on multiple processors and nodes, making them ideal for distributed computing.

        - **Custom Code**: Implementing multi-node support from scratch requires significant development effort, including managing communication between nodes and optimizing memory access patterns.
        - **Optimized Libraries**: Many versions of BLAS and LAPACK, such as those provided by OpenMPI or Intel MKL, support multi-node execution, making it easy to scale your code across multiple processors.

        In summary, using libraries like BLAS and LAPACK saves development time, ensures the accuracy of results, and significantly boosts performance, especially when scaling to large problems in an HPC setting.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using BLAS and LAPACK Across Multiple Nodes

        In large-scale HPC systems, leveraging multiple nodes can significantly reduce computation time. Many implementations of BLAS and LAPACK, such as Intel's MKL and OpenMPI, provide support for distributing tasks across multiple nodes in a cluster.

        ### How to Use BLAS and LAPACK with Multiple Nodes

        When running BLAS and LAPACK operations on multiple nodes, you typically rely on MPI (Message Passing Interface) to manage communication between nodes. Here's how it works:

        1. **MPI for Parallel Execution**: MPI is used to distribute matrix data across nodes. Each node will handle a portion of the matrix, and BLAS or LAPACK routines are used to perform the calculations locally. The results are then communicated back to the master node.

        2. **Scalability**: As the matrix size grows, distributing computations across nodes allows you to process larger datasets more quickly. This is particularly beneficial for tasks like matrix multiplication and solving linear systems.

        3. **Load Balancing**: Libraries like ScaLAPACK (a parallelized version of LAPACK) ensure that workloads are evenly distributed across nodes, optimizing the overall computation time.

        ### Example: Running BLAS on Multiple Nodes

        Below is an example of how to modify your code to run on multiple nodes using MPI. We'll use `MPI_Init` to initialize MPI and `MPI_Finalize` to clean up at the end.

        """
    )
    return


@app.cell
def _(os, subprocess):
    c_program_1 = '\n#include <mpi.h>\n#include <stdio.h>\n#include <stdlib.h>\n\nextern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,\n                   double *alpha, double *A, int *lda, double *B, int *ldb,\n                   double *beta, double *C, int *ldc);\n\nvoid print_matrix(const char* name, double *matrix, int rows, int cols) {\n    printf("%s:\\n", name);\n    for (int i = 0; i < rows; i++) {\n        for (int j = 0; j < cols; j++) {\n            printf("%f ", matrix[i * cols + j]);\n        }\n        printf("\\n");\n    }\n}\n\nint main(int argc, char** argv) {\n    MPI_Init(&argc, &argv);\n\n    int world_size;\n    MPI_Comm_size(MPI_COMM_WORLD, &world_size);\n\n    int world_rank;\n    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);\n\n    double A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};\n    double B[9] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};\n    double C[9];\n\n    int m = 3, n = 3, k = 3;\n    double alpha = 1.0, beta = 0.0;\n\n    if (world_rank == 0) {\n        dgemm_("N", "N", &m, &n, &k, &alpha, A, &m, B, &n, &beta, C, &m);\n        print_matrix("Matrix C (A * B)", C, m, n);\n    }\n\n    MPI_Barrier(MPI_COMM_WORLD);\n    MPI_Finalize();\n    return 0;\n}\n'
    c_filename_1 = 'mpi_blas_example.c'
    with open(c_filename_1, 'w') as c_file_1:
        c_file_1.write(c_program_1)
    print(f'C program written to {c_filename_1}')
    compile_command_1 = 'mpicc -o mpi_blas_example mpi_blas_example.c -lmkl_rt'
    compile_process_1 = subprocess.run(compile_command_1, shell=True, capture_output=True, text=True)
    if compile_process_1.returncode == 0:
        os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
        run_command_1 = 'mpirun -np 4 ./mpi_blas_example'
        run_process_1 = subprocess.run(run_command_1, shell=True, capture_output=True, text=True)
        if run_process_1.returncode == 0:
            print(f'Program Output:\n{run_process_1.stdout}')
        else:
            print(f'Program Error:\n{run_process_1.stderr}')
    else:
        print(f'Compilation failed:\n{compile_process_1.stderr}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.1 Introduction to Parallel I/O in HPC

        Parallel I/O is essential in High-Performance Computing (HPC) environments, especially when dealing with large datasets. In typical serial I/O operations, a single process reads or writes data, creating bottlenecks as file sizes grow. In contrast, **Parallel I/O** allows multiple processes to perform I/O operations concurrently, which significantly increases the performance of data-intensive applications.

        Parallel I/O is especially useful when combined with parallel filesystems like **Lustre** or **GPFS**. These filesystems are specifically designed to allow many processes to read and write large amounts of data simultaneously, distributing I/O operations across multiple storage devices to provide high throughput and scalability.

        In this section, we will demonstrate parallel I/O using **C** and **MPI**, where multiple processes write their data to a shared file concurrently and read the data back in parallel.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.2 Understanding Parallel Filesystems

        Parallel filesystems, such as **Lustre**, **GPFS**, or **BeeGFS**, are designed for use in HPC environments to manage large-scale data operations. They distribute data across multiple storage devices, enabling multiple processes to read and write data simultaneously.

        ### Key Features of Parallel Filesystems:
        - **High throughput**: Achieved by distributing data across multiple storage servers.
        - **Scalability**: Able to handle large datasets and a high number of concurrent processes.
        - **Redundancy**: Data is often stored redundantly across multiple disks to prevent data loss in case of hardware failure.
        - **Concurrent Access**: Multiple processes can access the same file at the same time, significantly improving performance in distributed applications.

        In a typical scenario, an application running on hundreds or thousands of compute nodes in a supercomputer accesses data stored in a parallel filesystem. Instead of waiting for one process to finish its I/O operation, each process can handle its I/O independently and concurrently, improving the overall performance.

        """
    )
    return


@app.cell
def _():
    c_code = '\n#include <stdio.h>\n#include <stdlib.h>\n#include <mpi.h>\n\nint main(int argc, char **argv) {\n    MPI_Init(&argc, &argv);  // Initialize MPI environment\n\n    int rank, size;\n    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank\n    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get number of processes\n\n    // Create a large array filled with the rank number\n    int N = 1000000;  // Size of the array\n    int *data = (int*) malloc(N * sizeof(int));\n    for (int i = 0; i < N; i++) {\n        data[i] = rank;\n    }\n\n    // Open a shared file for writing\n    MPI_File fh;\n    MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);\n\n    // Each process writes its data at the correct offset in the file\n    MPI_File_write_at_all(fh, rank * N * sizeof(int), data, N, MPI_INT, MPI_STATUS_IGNORE);\n    \n    // Close the file after writing\n    MPI_File_close(&fh);\n\n    // Synchronize all processes\n    MPI_Barrier(MPI_COMM_WORLD);\n\n    // Allocate space for reading back the data\n    int *read_data = (int*) malloc(N * sizeof(int));\n\n    // Open the file again for reading\n    MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);\n\n    // Each process reads its data back from the file\n    MPI_File_read_at_all(fh, rank * N * sizeof(int), read_data, N, MPI_INT, MPI_STATUS_IGNORE);\n\n    // Close the file after reading\n    MPI_File_close(&fh);\n\n    // Verify by printing the first and last elements of the read data\n    printf("Process %d: First element = %d, Last element = %d\\n", rank, read_data[0], read_data[N-1]);\n\n    // Free allocated memory\n    free(data);\n    free(read_data);\n\n    // Finalize the MPI environment\n    MPI_Finalize();\n    return 0;\n}\n'
    c_filename_2 = 'parallel_io_example.c'
    with open(c_filename_2, 'w') as c_file_2:
        c_file_2.write(c_code)
    print(f'C program written to {c_filename_2}')
    return


@app.cell
def _(subprocess):
    subprocess.run('# Step 1: Compile the C program using mpicc\nmpicc -o parallel_io_example parallel_io_example.c\n\n# Step 2: Run the compiled program with multiple processes using mpirun\nmpirun -np 4 ./parallel_io_example', shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.3 Explanation of the Parallel I/O Code

        In this code example, we use **MPI** (Message Passing Interface) to demonstrate how parallel I/O works. Here’s a breakdown of the key operations:

        1. **MPI Initialization**:
           The `MPI_Init()` function initializes the MPI environment, enabling communication between multiple processes running on different nodes.

        2. **Process Rank and Size**:
           - `MPI_Comm_rank()` retrieves the rank (ID) of each process.
           - `MPI_Comm_size()` retrieves the total number of processes involved.

        3. **Array Creation**:
           Each process creates a large array filled with its own rank number. This simulates data generation on each process that will be written to a shared file.

        4. **Parallel Write Operation**:
           - `MPI_File_open()` opens the file in **write mode** (`MPI_MODE_CREATE | MPI_MODE_WRONLY`).
           - `MPI_File_write_at_all()` is used to perform a parallel write. Each process writes its portion of the data at a specific offset, ensuring no overlap between processes' data.

        5. **Barrier Synchronization**:
           After the write operation, `MPI_Barrier()` ensures that all processes finish writing before proceeding to the next step.

        6. **Parallel Read Operation**:
           - `MPI_File_open()` reopens the file in **read mode** (`MPI_MODE_RDONLY`).
           - `MPI_File_read_at_all()` allows each process to read its portion of the data in parallel from the shared file.

        7. **Data Verification**:
           Each process prints the first and last elements of the data it read from the file. This verifies that the data was written and read correctly in parallel.

        ### Advantages:
        - **Efficiency**: By allowing concurrent I/O operations, we significantly reduce the time required to perform large-scale data reads and writes.
        - **Scalability**: The code can be scaled up to hundreds or thousands of processes, leveraging the full potential of parallel filesystems.
        - **No Bottlenecks**: Since each process performs I/O independently, there are no bottlenecks caused by sequential file access, making the solution ideal for large-scale HPC applications.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6.1 Introduction to Performance Tuning and Analysis

        In High-Performance Computing (HPC), performance tuning is a critical step for ensuring that applications run as efficiently as possible. With large-scale computations, identifying and resolving bottlenecks in code can lead to substantial performance improvements.

        Performance tuning typically involves:

        1. **Profiling**: This involves identifying bottlenecks by analyzing where the application spends most of its time.
        2. **Optimization**: Applying various optimizations, such as reducing memory usage, improving I/O operations, or parallelizing parts of the code.
        3. **Reprofiling**: After optimizations are applied, reprofile the application to assess the impact of the changes and iterate if necessary.

        We will use tools such as `gprof`, `perf`, and `Intel VTune` to analyze the performance of a matrix computation in C with MPI.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6.2 Profiling with Standard HPC Tools

        In this exercise, we will create and compile a C program that performs matrix multiplication using MPI. We will then profile this program using **gprof** and **perf**, two widely used profiling tools in HPC.

        - **gprof**: A GNU profiler that shows the time spent in each function and helps pinpoint performance bottlenecks.
        - **perf**: A performance monitoring tool that provides detailed reports on CPU cycles, cache misses, and other hardware events.

        ### Step 1: Writing the Matrix Multiplication Code

        First, we will write a simple C program that performs matrix multiplication using MPI. The program will create matrices, distribute work among multiple processes, and combine the results.

        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%writefile mpi_matrix_multiply.c
    # 
    # #include <mpi.h>
    # #include <stdio.h>
    # #include <stdlib.h>
    # 
    # // Function to perform matrix multiplication
    # void matrix_multiply(int n, double* A, double* B, double* C) {
    #     for (int i = 0; i < n; i++) {
    #         for (int j = 0; j < n; j++) {
    #             C[i*n + j] = 0.0;
    #             for (int k = 0; k < n; k++) {
    #                 C[i*n + j] += A[i*n + k] * B[k*n + j];
    #             }
    #         }
    #     }
    # }
    # 
    # int main(int argc, char** argv) {
    #     MPI_Init(&argc, &argv);
    # 
    #     int world_size;
    #     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    # 
    #     int world_rank;
    #     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    # 
    #     int n = 1000;  // Matrix size
    #     double *A, *B, *C;
    # 
    #     if (world_rank == 0) {
    #         // Allocate memory for matrices A, B, and C
    #         A = (double*) malloc(n * n * sizeof(double));
    #         B = (double*) malloc(n * n * sizeof(double));
    #         C = (double*) malloc(n * n * sizeof(double));
    # 
    #         // Initialize matrices A and B
    #         for (int i = 0; i < n*n; i++) {
    #             A[i] = rand() % 100;
    #             B[i] = rand() % 100;
    #         }
    #     }
    # 
    #     // Perform matrix multiplication on rank 0 process
    #     if (world_rank == 0) {
    #         matrix_multiply(n, A, B, C);
    #         printf("Matrix multiplication completed.\n");
    #     }
    # 
    #     MPI_Finalize();
    # 
    #     if (world_rank == 0) {
    #         free(A);
    #         free(B);
    #         free(C);
    #     }
    # 
    #     return 0;
    # }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6.3 Compilation and Profiling the Program

        We will now compile the program using `mpicc` (MPI C Compiler) and then profile it using `gprof` and `perf`. 

        ### Step 2: Compile the Program

        Use `mpicc` to compile the matrix multiplication program:

        The -pg flag enables profiling for gprof.

        ### Step 3: Run the Program
        Next, we will run the program using mpirun:

        ### Step 4: Profiling with gprof
        After the program runs, you can generate a profiling report with gprof:

        """
    )
    return


@app.cell
def _(subprocess):
    compile_process_2 = subprocess.run('mpicc -pg -o mpi_matrix_multiply mpi_matrix_multiply.c', shell=True, capture_output=True, text=True)
    if compile_process_2.returncode == 0:
        print('Compilation successful.')
        run_command_2 = 'mpirun --oversubscribe -np 4 ./mpi_matrix_multiply'
        run_process_2 = subprocess.run(run_command_2, shell=True, capture_output=True, text=True)
        if run_process_2.returncode == 0:
            print('Program Output:')
            print(run_process_2.stdout)
        else:
            print(f'Program Error:\n{run_process_2.stderr}')
        gprof_command = 'gprof ./mpi_matrix_multiply gmon.out > analysis.txt'
        subprocess.run(gprof_command, shell=True)
        with open('analysis.txt', 'r') as file:
            analysis_output = file.read()
            print('Gprof Analysis Output:')
            print(analysis_output)
        perf_command = 'perf stat -d mpirun --oversubscribe -np 4 ./mpi_matrix_multiply'
        perf_process = subprocess.run(perf_command, shell=True, capture_output=True, text=True)
        if perf_process.returncode == 0:
            print('Perf Output:')
            print(perf_process.stdout)
        else:
            print(f'Perf Error:\n{perf_process.stderr}')
    else:
        print(f'Compilation failed:\n{compile_process_2.stderr}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Understanding and Interpreting Performance Profiling

        In this section, we will compile, run, and profile an MPI-based matrix multiplication program using `gprof` and `perf`.

        #### Step-by-Step Process:

        1. **Compilation with Profiling Enabled**:
           - The program is compiled using `mpicc` with the `-pg` flag to enable profiling for `gprof`.
           - This generates a binary (`mpi_matrix_multiply`) and enables the collection of profiling data during execution.

        2. **Execution with `mpirun`**:
           - The program is executed using `mpirun` with 4 processes and the `--oversubscribe` option. Oversubscription allows more processes than available CPUs to run.
           - The matrix multiplication completes, and profiling data is generated in a `gmon.out` file.

        3. **Profiling with `gprof`**:
           - `gprof` analyzes the execution of the program and provides a breakdown of where time is being spent.
           - The `analysis.txt` file contains detailed profiling data, including:
             - **Flat Profile**: Shows the time spent in each function. Use this to identify the most time-consuming functions.
             - **Call Graph**: Displays the call hierarchy and the time spent in both the parent and child functions. This helps understand which functions call other functions and their time cost.

        4. **System-Level Profiling with `perf`**:
           - `perf` provides hardware-level insights into CPU usage, cache hits, and system resource utilization during the execution of the program.
           - Key metrics to interpret from `perf`:
             - **task-clock**: Total time spent on the task.
             - **context-switches**: Number of context switches during execution.
             - **page-faults**: Number of memory page faults.
             - **cycles/instructions/branches**: These metrics are crucial for understanding how efficiently the program is running on the CPU (if supported by the system).

        #### How to Interpret the Output:

        - **`gprof` Output**:
          - Identify the functions with the highest time consumption. These are the bottlenecks where optimization efforts should focus.
          - Review the call graph to see how functions interact and whether any redundant calls can be optimized.

        - **`perf` Output**:
          - High context switches or page faults may indicate inefficient resource management.
          - The `cycles` and `instructions` counters (if supported) help assess CPU efficiency. A high ratio of instructions per cycle is desirable.

        By combining the insights from both `gprof` and `perf`, you can target both function-level and system-level optimizations to improve the performance of your MPI program.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8. Applying Optimizations

        Once bottlenecks are identified, the next step is to apply optimizations. In this section, we will optimize matrix operations using techniques such as loop unrolling, vectorization, and memory access optimization.

        ### 8.1 Loop Unrolling and Vectorization

        We will revisit loop unrolling and vectorization to see how they can improve performance in matrix operations.

        """
    )
    return


@app.cell
def _(random, time):
    def create_matrix(rows, cols):
        return [[random.random() for _ in range(cols)] for _ in range(rows)]

    def basic_matrix_sum(matrix):
        total = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                total = total + matrix[i][j]
        return total

    def vectorized_matrix_sum(matrix):
        return sum((sum(row) for row in matrix))
    matrix_1 = create_matrix(4000, 4000)
    start_time_1 = time.time()
    basic_sum = basic_matrix_sum(matrix_1)
    print('Basic matrix sum time:', time.time() - start_time_1)
    start_time_1 = time.time()
    vectorized_sum = vectorized_matrix_sum(matrix_1)
    print('Vectorized matrix sum time:', time.time() - start_time_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        This example compares the performance of a basic loop-based matrix sum with a vectorized version using NumPy's built-in `sum` function. Vectorization allows for faster computation by leveraging SIMD instructions.

        ### Exercise:

        Try optimizing the `basic_matrix_sum` function by manually unrolling the loops. Measure the performance impact and compare it with the vectorized approach.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 9. Memory Access Optimization and Cache Utilization

        Memory access patterns greatly affect the performance of HPC applications. In this section, we'll explore techniques to optimize memory access and improve cache utilization.

        ### 9.1 Optimizing Memory Access Patterns

        Efficient memory access patterns reduce cache misses, leading to faster execution times. We'll analyze the impact of row-major vs. column-major access.

        """
    )
    return


@app.cell
def _(random, time):
    def create_matrix_1(rows, cols):
        return [[random.random() for _ in range(cols)] for _ in range(rows)]

    def row_major_sum(matrix):
        total = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                total = total + matrix[i][j]
        return total

    def column_major_sum(matrix):
        total = 0
        for j in range(len(matrix[0])):
            for i in range(len(matrix)):
                total = total + matrix[i][j]
        return total
    matrix_2 = create_matrix_1(3000, 3000)
    start_time_2 = time.time()
    row_sum = row_major_sum(matrix_2)
    print('Row-major sum time:', time.time() - start_time_2)
    start_time_2 = time.time()
    column_sum = column_major_sum(matrix_2)
    print('Column-major sum time:', time.time() - start_time_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        The example above compares row-major and column-major memory access patterns. Typically, row-major access is faster on most systems because it aligns better with how data is stored in memory.

        ### Exercise:

        Modify the code to measure the cache hit rate (if possible using advanced profiling tools or libraries) for each access pattern. Observe how different matrix sizes affect cache utilization and performance.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 10. Leveraging High-Performance Libraries

        Using specialized HPC libraries can significantly enhance the performance of your applications. This section explores how to use BLAS, LAPACK, and other optimized libraries in your code.

        ### 10.1 Using BLAS and LAPACK for Matrix Operations

        BLAS (Basic Linear Algebra Subprograms) and LAPACK are standard libraries providing highly optimized implementations of basic linear algebra routines.

        """
    )
    return


@app.cell
def _(random):
    def create_matrix_3x3():
        return [[random.random() for _ in range(3)] for _ in range(3)]

    def matrix_multiply_1(A, B):
        C = [[0.0 for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i][j] = C[i][j] + A[i][k] * B[k][j]
        return C

    def lu_decomposition(A):
        n = len(A)
        LU = [row[:] for row in A]
        piv = list(range(n))
        for k in range(n):
            pivot_value = LU[k][k]
            if pivot_value == 0:
                raise ValueError('Matrix is singular')
            for i in range(k + 1, n):
                LU[i][k] = LU[i][k] / pivot_value
                for j in range(k + 1, n):
                    LU[i][j] = LU[i][j] - LU[i][k] * LU[k][j]
        return (LU, piv)

    def invert_matrix(LU):
        n = len(LU)
        inv_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            inv_matrix[i][i] = 1.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    inv_matrix[i][j] = -LU[i][j]
        return inv_matrix
    A_1 = create_matrix_3x3()
    B_1 = create_matrix_3x3()
    C = matrix_multiply_1(A_1, B_1)
    LU, piv = lu_decomposition(A_1)
    inv_matrix = invert_matrix(LU)
    print('Matrix A:')
    for row in A_1:
        print(row)
    print('\nMatrix B:')
    for row in B_1:
        print(row)
    print('\nResult of matrix multiplication (A * B = C):')
    for row in C:
        print(row)
    print('\nMatrix inversion of A using LU decomposition:')
    for row in inv_matrix:
        print(row)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        This example demonstrates how to use the BLAS `dgemm` function for matrix multiplication and the LAPACK `dgetrf` function for matrix inversion. These libraries are optimized for performance on many HPC systems.

        ### Exercise:

        Try using other functions from BLAS and LAPACK, such as `dsymv` for symmetric matrix-vector multiplication or `dgeev` for eigenvalue computation. Compare the performance of these library functions with your custom implementations.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 11. Advanced Performance Tuning with Parallel I/O

        Efficient I/O operations are critical for handling large datasets in HPC applications. This section covers advanced parallel I/O techniques using mpi4py.

        ### 11.1 Implementing Parallel I/O

        We will extend our previous examples by implementing collective I/O operations, which can be more efficient for large-scale data processing.

        """
    )
    return


@app.cell
def _():
    c_code_1 = '\n#include <mpi.h>\n#include <stdio.h>\n#include <stdlib.h>\n\nint main(int argc, char** argv) {\n    MPI_Init(&argc, &argv);  // Initialize the MPI environment\n\n    MPI_Comm comm = MPI_COMM_WORLD;\n    int rank, size;\n    MPI_Comm_rank(comm, &rank);  // Get the rank of the process\n    MPI_Comm_size(comm, &size);  // Get the total number of processes\n\n    // Create a large array on each process, filled with the rank number\n    int data_size = 1000000;\n    int* data = (int*)malloc(data_size * sizeof(int));\n    for (int i = 0; i < data_size; i++) {\n        data[i] = rank;\n    }\n\n    // Write data collectively to a shared file\n    MPI_File fh;\n    MPI_File_open(comm, "collective_output.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);\n    MPI_File_write_at_all(fh, rank * data_size * sizeof(int), data, data_size, MPI_INT, MPI_STATUS_IGNORE);\n    MPI_File_close(&fh);  // Close the file after writing\n\n    // Allocate memory for reading the data back\n    int* collected_data = (int*)malloc(data_size * sizeof(int));\n\n    // Reading data collectively from the shared file\n    MPI_File_open(comm, "collective_output.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);\n    MPI_File_read_at_all(fh, rank * data_size * sizeof(int), collected_data, data_size, MPI_INT, MPI_STATUS_IGNORE);\n    MPI_File_close(&fh);  // Close the file after reading\n\n    // Print out a summary of the data to verify the read operation\n    printf("Process %d: First element = %d, Last element = %d\\n", rank, collected_data[0], collected_data[data_size - 1]);\n\n    // Free dynamically allocated memory\n    free(data);\n    free(collected_data);\n\n    MPI_Finalize();  // Finalize the MPI environment\n    return 0;\n}\n'
    with open('mpi_collective_io.c', 'w') as file_1:
        file_1.write(c_code_1)
    print('C code written to mpi_collective_io.c')
    return


@app.cell
def _(subprocess):
    subprocess.run('# Load MPI module (if required by your environment)\nmodule load openmpi/4.0.3\n\n# Compile the C code\nmpicc -o mpi_collective_io mpi_collective_io.c', shell=True)
    return


@app.cell
def _(subprocess):
    subprocess.run('# Run the program with oversubscription if necessary\nmpirun --oversubscribe -np 4 ./mpi_collective_io', shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        In this example, each MPI process writes and reads a portion of data from a shared file using collective I/O operations. This technique improves the efficiency of data handling in parallel applications.

        ### Exercise:

        Modify the code to test the performance impact of different file access modes, such as `MPI.MODE_APPEND` or non-collective I/O. Analyze how these changes affect the scalability of I/O operations when running on multiple processes.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Comprehensive Performance Analysis and Tuning

        In this section, we perform a comprehensive performance analysis and tuning of a complex HPC (High-Performance Computing) application. The focus is on profiling, optimization, and parallel I/O techniques to improve the performance of scientific computations. We will use the example code provided to illustrate these concepts.

        ### 12.1 Case Study: Performance Tuning of a Scientific Application

        We apply profiling, optimization, and parallel I/O techniques to a real-world scientific computation involving matrix operations and parallel I/O. The provided code examples demonstrate matrix multiplication and inversion using BLAS and LAPACK routines, as well as parallel I/O with MPI.

        #### Code Explanation

        ##### `mpi_blas_lapack.c`

        This code performs matrix operations and parallel I/O using MPI.

        1. **Matrix Computation**: 
           - The `optimized_computation` function performs matrix multiplication (`dgemm_`) and matrix inversion (`dgetrf_` and `dgetri_`) using BLAS and LAPACK routines.
           - It initializes matrices `A` and `B`, performs the operations, and computes the sum of elements in the inverted matrix.

        2. **Parallel I/O**:
           - The `main` function initializes MPI, performs matrix computations, and saves the result to a file using MPI I/O.
           - Each MPI process writes its result to a shared file, ensuring proper parallel file access.

        ##### `blas_lapack_example.c`

        This code demonstrates basic usage of BLAS and LAPACK routines.

        1. **Matrix Multiplication**:
           - The `dgemm_` function multiplies two matrices `A` and `B` and stores the result in matrix `C`.
           - The result is printed to the console.

        2. **Solving Linear Systems**:
           - The `dgesv_` function solves a system of linear equations `A * X = B`.
           - The solution is printed to the console.

        #### Compilation and Execution

        The programs are compiled using the Intel Math Kernel Library (IMKL) and OpenMPI, and executed with MPI to leverage parallel processing capabilities.

        ```bash
        # Load necessary modules (IMKL and OpenMPI)
        module load imkl/2020.1.217
        module load openmpi/4.0.3

        # Compile the C program
        mpicc -o blas_lapack_example blas_lapack_example.c -lmkl_rt

        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%writefile blas_lapack_example.c
    # #include <stdio.h>
    # #include <stdlib.h>
    # 
    # // Declare BLAS and LAPACK routines
    # extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
    #                    double *alpha, double *A, int *lda, double *B, int *ldb,
    #                    double *beta, double *C, int *ldc);
    # 
    # extern void dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv,
    #                    double *B, int *ldb, int *info);
    # 
    # void print_matrix(const char* name, double *matrix, int rows, int cols) {
    #     printf("%s:\n", name);
    #     for (int i = 0; i < rows; i++) {
    #         for (int j = 0; j < cols; j++) {
    #             printf("%f ", matrix[i * cols + j]);
    #         }
    #         printf("\n");
    #     }
    # }
    # 
    # int main() {
    #     // Example: Matrix multiplication using dgemm (BLAS)
    #     
    #     // Matrices A (3x3), B (3x3), and C (3x3) for A * B = C
    #     double A[9] = {1.0, 2.0, 3.0, 
    #                    4.0, 5.0, 6.0, 
    #                    7.0, 8.0, 9.0};
    #     double B[9] = {9.0, 8.0, 7.0, 
    #                    6.0, 5.0, 4.0, 
    #                    3.0, 2.0, 1.0};
    #     double C[9];
    #     
    #     int m = 3, n = 3, k = 3;  // Dimensions of matrices
    #     double alpha = 1.0, beta = 0.0;
    #     
    #     // Matrix multiplication C = alpha * A * B + beta * C
    #     dgemm_("N", "N", &m, &n, &k, &alpha, A, &m, B, &n, &beta, C, &m);
    #     
    #     // Print the result of A * B
    #     print_matrix("Matrix C (A * B)", C, m, n);
    # 
    #     // Example: Solving a linear system using dgesv (LAPACK)
    #     
    #     // A (3x3) and B (3x1), solve A * X = B
    #     double A2[9] = {3.0, 2.0, -1.0,
    #                     2.0, -2.0, 4.0,
    #                     -1.0, 0.5, -1.0};
    #     double B2[3] = {1.0, -2.0, 0.0};  // Right-hand side
    #     
    #     int ipiv[3];  // Pivot indices
    #     int info;     // Return info
    #     int nrhs = 1; // Number of right-hand sides
    #     
    #     // Solve the system of equations A * X = B
    #     dgesv_(&m, &nrhs, A2, &m, ipiv, B2, &m, &info);
    #     
    #     if (info == 0) {
    #         // Print the solution
    #         print_matrix("Solution to A * X = B", B2, m, nrhs);
    #     } else {
    #         printf("An error occurred: dgesv returned info = %d\n", info);
    #     }
    # 
    #     return 0;
    # }
    return


@app.cell
def _(subprocess):
    subprocess.run('# Load necessary modules (IMKL and OpenMPI)\nmodule load imkl/2020.1.217\nmodule load openmpi/4.0.3\n\n# Compile the C program\nmpicc -o blas_lapack_example blas_lapack_example.c -lmkl_rt', shell=True)
    return


@app.cell
def _(subprocess):
    subprocess.run('# Run the program with 4 processes (oversubscription enabled if necessary)\nmpirun --oversubscribe -np 4 ./blas_lapack_example', shell=True)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%writefile mpi_blas_lapack_profiling.c
    # #include <stdio.h>
    # #include <stdlib.h>
    # #include <mpi.h>
    # #include <sys/time.h>
    # 
    # // External declarations for BLAS and LAPACK routines
    # extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
    #                    double *alpha, double *A, int *lda, double *B, int *ldb,
    #                    double *beta, double *C, int *ldc);
    # 
    # extern void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
    # extern void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work, int *lwork, int *info);
    # 
    # // Function to measure elapsed time
    # double get_elapsed_time(struct timeval *start, struct timeval *end) {
    #     return ((end->tv_sec - start->tv_sec) + (end->tv_usec - start->tv_usec) / 1.0e6);
    # }
    # 
    # // Function to perform matrix multiplication and inversion
    # double optimized_computation(double *A, double *B, int N) {
    #     struct timeval start, end;
    #     gettimeofday(&start, NULL);
    # 
    #     int m = N, n = N, k = N;
    #     double alpha = 1.0, beta = 0.0;
    #     double *C = (double *)malloc(N * N * sizeof(double));
    # 
    #     // Perform matrix multiplication C = A * B using BLAS
    #     dgemm_("N", "N", &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
    # 
    #     // Perform LU factorization using LAPACK
    #     int *ipiv = (int *)malloc(N * sizeof(int));
    #     int info;
    #     dgetrf_(&N, &N, C, &N, ipiv, &info);
    # 
    #     if (info != 0) {
    #         printf("LU factorization failed with info = %d\n", info);
    #         free(C);
    #         free(ipiv);
    #         return -1;
    #     }
    # 
    #     // Compute matrix inverse using LAPACK
    #     int lwork = N * N;
    #     double *work = (double *)malloc(lwork * sizeof(double));
    #     dgetri_(&N, C, &N, ipiv, work, &lwork, &info);
    # 
    #     if (info != 0) {
    #         printf("Matrix inversion failed with info = %d\n", info);
    #         free(C);
    #         free(ipiv);
    #         free(work);
    #         return -1;
    #     }
    # 
    #     // Sum the inverted matrix elements
    #     double sum = 0.0;
    #     for (int i = 0; i < N * N; i++) {
    #         sum += C[i];
    #     }
    # 
    #     gettimeofday(&end, NULL);
    #     printf("Computation time: %f seconds\n", get_elapsed_time(&start, &end));
    # 
    #     free(C);
    #     free(ipiv);
    #     free(work);
    #     return sum;
    # }
    # 
    # // Main function to perform computation and save the result using MPI I/O
    # int main(int argc, char **argv) {
    #     MPI_Init(&argc, &argv);
    # 
    #     int rank, size;
    #     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #     MPI_Comm_size(MPI_COMM_WORLD, &size);
    # 
    #     int N = 500;  // Matrix size
    #     double *A = (double *)malloc(N * N * sizeof(double));
    #     double *B = (double *)malloc(N * N * sizeof(double));
    # 
    #     // Initialize matrices A and B with random data
    #     srand(rank + 1);  // Seed based on rank
    #     for (int i = 0; i < N * N; i++) {
    #         A[i] = rand() / (double)RAND_MAX;
    #         B[i] = rand() / (double)RAND_MAX;
    #     }
    # 
    #     // Perform optimized computation
    #     double result = optimized_computation(A, B, N);
    # 
    #     // Parallel I/O to save results
    #     MPI_File file_handle;
    #     MPI_File_open(MPI_COMM_WORLD, "final_result.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file_handle);
    #     
    #     // Save result from each process into the file
    #     MPI_File_write_at(file_handle, rank * sizeof(double), &result, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
    #     MPI_File_close(&file_handle);
    # 
    #     printf("Process %d completed its task and saved the result: %f\n", rank, result);
    # 
    #     free(A);
    #     free(B);
    #     MPI_Finalize();
    #     return 0;
    # }
    return


app._unparsable_cell(
    r"""
    # Load necessary modules (IMKL and OpenMPI) and compile the C program
    !module load imkl/2020.1.217
    !module load openmpi/4.0.3
    !mpicc -o mpi_blas_lapack_profiling mpi_blas_lapack_profiling.c -lmkl_rt

    # Run the program with 4 processes (oversubscription enabled if necessary)
    !mpirun --oversubscribe -np 4 ./mpi_blas_lapack_profiling
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Optimizing Code
        After profiling, you can apply optimizations based on the identified bottlenecks. Here are some common optimization strategies:

        Matrix Operations Optimization
        Optimize BLAS/LAPACK Calls: Ensure that you are using the most efficient BLAS and LAPACK routines for your hardware. For example, use Intel MKL or OpenBLAS optimized libraries.

        Data Locality: Ensure that matrices and other large data structures are accessed in a cache-friendly manner to reduce cache misses.

        Parallel I/O Optimization
        Reduce I/O Contention: Use collective I/O operations where possible to minimize I/O contention between processes.

        Optimize File Access: Ensure that file access patterns are optimized for parallel writing, and use MPI I/O hints to improve performance.

        Code Example for Matrix Multiplication Optimization


        ```c
        // Example of optimizing matrix multiplication by improving cache usage
        void optimized_dgemm(char transa, char transb, int m, int n, int k, double alpha,
                             double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
            // Use a block size for better cache utilization
            int block_size = 64; // Adjust this based on your cache size
            for (int i = 0; i < m; i += block_size) {
                for (int j = 0; j < n; j += block_size) {
                    for (int k = 0; k < k; k += block_size) {
                        // Perform block matrix multiplication
                        // Ensure that you do not go out of bounds
                        int m1 = min(block_size, m - i);
                        int n1 = min(block_size, n - j);
                        int k1 = min(block_size, k - k);
                        dgemm_(&transa, &transb, &m1, &n1, &k1, &alpha, &A[i * lda + k], &lda,
                               &B[k * ldb + j], &ldb, &beta, &C[i * ldc + j], &ldc);
                    }
                }
            }
        }

        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%writefile mpi_blas_lapack_optimised.c
    # #include <stdio.h>
    # #include <stdlib.h>
    # #include <mpi.h>
    # #include <sys/time.h>
    # 
    # // External declarations for BLAS and LAPACK routines
    # extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
    #                    double *alpha, double *A, int *lda, double *B, int *ldb,
    #                    double *beta, double *C, int *ldc);
    # 
    # extern void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
    # extern void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work, int *lwork, int *info);
    # 
    # // Function to measure elapsed time
    # double get_elapsed_time(struct timeval *start, struct timeval *end) {
    #     return ((end->tv_sec - start->tv_sec) + (end->tv_usec - start->tv_usec) / 1.0e6);
    # }
    # 
    # // Optimized matrix multiplication using blocking for better cache utilization
    # void optimized_dgemm(char transa, char transb, int m, int n, int k, double alpha,
    #                      double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
    #     int block_size = 64; // Adjust this based on your cache size
    #     for (int i = 0; i < m; i += block_size) {
    #         for (int j = 0; j < n; j += block_size) {
    #             for (int l = 0; l < k; l += block_size) {
    #                 // Determine the dimensions of the current block
    #                 int m1 = (i + block_size > m) ? (m - i) : block_size;
    #                 int n1 = (j + block_size > n) ? (n - j) : block_size;
    #                 int l1 = (l + block_size > k) ? (k - l) : block_size;
    # 
    #                 // Perform block matrix multiplication
    #                 dgemm_(&transa, &transb, &m1, &n1, &l1, &alpha, &A[i + l * lda], &lda,
    #                        &B[l + j * ldb], &ldb, &beta, &C[i + j * ldc], &ldc);
    #             }
    #         }
    #     }
    # }
    # 
    # 
    # // Function to perform matrix multiplication and inversion
    # double optimized_computation(double *A, double *B, int N) {
    #     struct timeval start, end;
    #     gettimeofday(&start, NULL);
    # 
    #     int m = N, n = N, k = N;
    #     double alpha = 1.0, beta = 0.0;
    #     double *C = (double *)malloc(N * N * sizeof(double));
    #     if (C == NULL) {
    #         perror("Failed to allocate memory for matrix C");
    #         return -1;
    #     }
    # 
    #     // Perform matrix multiplication C = A * B using the optimized BLAS function
    #     optimized_dgemm('N', 'N', m, n, k, alpha, A, m, B, k, beta, C, m);
    # 
    #     // Perform LU factorization using LAPACK
    #     int *ipiv = (int *)malloc(N * sizeof(int));
    #     if (ipiv == NULL) {
    #         perror("Failed to allocate memory for IPIV");
    #         free(C);
    #         return -1;
    #     }
    #     int info;
    #     dgetrf_(&N, &N, C, &N, ipiv, &info);
    # 
    #     if (info != 0) {
    #         printf("LU factorization failed with info = %d\n", info);
    #         free(C);
    #         free(ipiv);
    #         return -1;
    #     }
    # 
    #     // Compute matrix inverse using LAPACK
    #     int lwork = N * N;
    #     double *work = (double *)malloc(lwork * sizeof(double));
    #     if (work == NULL) {
    #         perror("Failed to allocate memory for work");
    #         free(C);
    #         free(ipiv);
    #         return -1;
    #     }
    #     dgetri_(&N, C, &N, ipiv, work, &lwork, &info);
    # 
    #     if (info != 0) {
    #         printf("Matrix inversion failed with info = %d\n", info);
    #         free(C);
    #         free(ipiv);
    #         free(work);
    #         return -1;
    #     }
    # 
    #     // Sum the inverted matrix elements
    #     double sum = 0.0;
    #     for (int i = 0; i < N * N; i++) {
    #         sum += C[i];
    #     }
    # 
    #     gettimeofday(&end, NULL);
    #     printf("Computation time: %f seconds\n", get_elapsed_time(&start, &end));
    # 
    #     free(C);
    #     free(ipiv);
    #     free(work);
    #     return sum;
    # }
    # 
    # // Main function to perform computation and save the result using MPI I/O
    # int main(int argc, char **argv) {
    #     MPI_Init(&argc, &argv);
    # 
    #     int rank, size;
    #     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #     MPI_Comm_size(MPI_COMM_WORLD, &size);
    # 
    #     int N = 500;  // Matrix size
    #     double *A = (double *)malloc(N * N * sizeof(double));
    #     double *B = (double *)malloc(N * N * sizeof(double));
    #     if (A == NULL || B == NULL) {
    #         perror("Failed to allocate memory for matrices A and B");
    #         MPI_Finalize();
    #         return -1;
    #     }
    # 
    #     // Initialize matrices A and B with random data
    #     srand(rank + 1);  // Seed based on rank
    #     for (int i = 0; i < N * N; i++) {
    #         A[i] = rand() / (double)RAND_MAX;
    #         B[i] = rand() / (double)RAND_MAX;
    #     }
    # 
    #     // Perform optimized computation
    #     double result = optimized_computation(A, B, N);
    # 
    #     // Parallel I/O to save results
    #     MPI_File file_handle;
    #     MPI_File_open(MPI_COMM_WORLD, "final_result.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file_handle);
    #     
    #     // Save result from each process into the file
    #     MPI_File_write_at(file_handle, rank * sizeof(double), &result, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
    #     MPI_File_close(&file_handle);
    # 
    #     printf("Process %d completed its task and saved the result: %f\n", rank, result);
    # 
    #     free(A);
    #     free(B);
    #     MPI_Finalize();
    #     return 0;
    # }
    return


app._unparsable_cell(
    r"""
    # Load necessary modules (IMKL and OpenMPI) and compile the C program
    !module load imkl/2020.1.217
    !module load openmpi/4.0.3
    !mpicc -o mpi_blas_lapack_optimised mpi_blas_lapack_optimised.c -lmkl_rt

    # Run the program with 4 processes (oversubscription enabled if necessary)
    !mpirun --oversubscribe -np 4 ./mpi_blas_lapack_optimised
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Optimization of MPI, BLAS, and LAPACK Computation

        ## Overview

        In this notebook, we implemented and optimized a matrix multiplication and inversion routine using MPI for parallel computation and BLAS/LAPACK libraries for efficient mathematical operations. The goal was to reduce computation time and improve performance by applying specific optimizations.

        ## Optimization Details

        ### 1. Optimized Matrix Multiplication Using Blocking

        **Original Approach:** 
        The standard `dgemm` function from BLAS was used for matrix multiplication. While effective, this approach does not always fully utilize the CPU cache, especially for large matrices.

        **Optimization Applied:**
        We implemented a custom version of matrix multiplication with blocking, which improves cache performance. Blocking divides the matrices into smaller sub-matrices or "blocks" that fit into the CPU cache. This reduces the number of cache misses and improves performance.

        **Why This Works:**

        Cache Efficiency: Blocking keeps data in cache, minimizing the need to repeatedly load data from slower memory.
        Reduced Cache Misses: By operating on smaller blocks, the number of cache misses is reduced, leading to faster computations.

        ### 2. Matrix Inversion and LU Factorization
        Approach Used:

        LU Factorization: We used LAPACK's dgetrf_ for LU decomposition.
        Matrix Inversion: We used dgetri_ to compute the inverse of the matrix.
        Optimization Considerations:

        Proper Memory Management: Allocated and freed memory appropriately to avoid memory leaks and ensure efficient use of resources.
        Error Handling: Included checks for failure in LU factorization and matrix inversion to handle potential issues gracefully.

        #### optional exercise:

        Expand the case study by adding more complex operations, such as eigenvalue computation or solving a system of linear equations. Profile and optimize these additional steps, and analyze how the performance scales with the problem size and number of processes.

        ### End of the practice.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

