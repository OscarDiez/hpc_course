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
    # Example: Profiling a simple matrix multiplication function using cProfile

    import cProfile
    import numpy as np

    def matrix_multiply(A, B):
        return np.dot(A, B)

    # Create large random matrices
    A = np.random.rand(1000, 1000)
    B = np.random.rand(1000, 1000)

    # Profile the matrix multiplication
    cProfile.run('matrix_multiply(A, B)')

    # The output will show where the time is being spent in the function
    return cProfile, np


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
def _(np):
    import time

    def sum_rows(matrix):
        total = 0
        for row in matrix:
            total = total + sum(row)
        return total

    def sum_columns(matrix):
        total = 0
        for col in range(matrix.shape[1]):
            total = total + sum(matrix[:, col])
        return total
    matrix = np.random.rand(10000, 10000)
    start_time = time.time()
    sum_rows(matrix)
    print('Row-wise sum time:', time.time() - start_time)
    start_time = time.time()
    sum_columns(matrix)
    print('Column-wise sum time:', time.time() - start_time)
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
        ## 4. High-Performance Libraries for Scientific Computing

        Leveraging high-performance libraries can save development time and ensure that your code is optimized for modern HPC architectures.

        ### 4.1 Using BLAS and LAPACK for Linear Algebra

        BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra Package) are standard libraries that provide optimized implementations of basic linear algebra routines.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Check if NumPy and SciPy are installed, and install if not
    import sys
    import subprocess

    def install(package):
        subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", package])

    # Try importing necessary packages and install if not available
    try:
    except ImportError:
        install('numpy')

    try:
        from scipy.linalg import blas
    except ImportError:
        install('scipy')

    # Now perform the matrix multiplication using BLAS

    # Create random matrices A and B
    A = np.random.rand(3, 3)
    B = np.random.rand(3, 3)

    # Ensure matrices are in Fortran-contiguous order
    A_f = np.asfortranarray(A)
    B_f = np.asfortranarray(B)

    # Using BLAS dgemm for matrix multiplication
    C = blas.dgemm(1.0, A_f, B_f)

    # Print the result
    print(\"Matrix A:\n\", A)
    print(\"Matrix B:\n\", B)
    print(\"Resulting matrix C (A * B):\n\", C)
    print(\"Resulting matrix shape:\", C.shape)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        Here, we use the `dgemm` function from BLAS, accessed via SciPy, to perform matrix multiplication. This function is highly optimized for performance on many HPC systems, often outperforming custom implementations.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 5 Checkpointing in High-Performance Computing (HPC)

        ## Introduction

        Checkpointing is an essential technique in HPC to ensure that long-running computations can recover from failures without having to start over from the beginning. It involves periodically saving the state of an application so that it can be resumed from the last checkpoint.

        In this notebook, we'll simulate a checkpointing mechanism using Python. We'll periodically save the state of a computation to a file, and if the program is interrupted, it can resume from the last saved state.

        ### Part 1: Setting Up the Environment

        Let's start by importing the necessary libraries and setting up the checkpointing mechanism.

        """
    )
    return


@app.cell
def _():
    import os
    import pickle

    def save_checkpoint(data, checkpoint_file='checkpoint.pkl'):
        """Save the current state to a checkpoint file."""
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved: {data}")

    def load_checkpoint(checkpoint_file='checkpoint.pkl'):
        """Load the state from a checkpoint file."""
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Checkpoint loaded: {data}")
            return data
        return None
    return load_checkpoint, save_checkpoint


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part 2: Simulating a Computation with Checkpointing

        We'll simulate a long-running computation, such as a loop that performs some calculations. We'll save the state of the computation at regular intervals (checkpoints). If the program stops, it can resume from the last checkpoint.

        ### Steps:
        1. **Run a computation** and save the state every few iterations.
        2. **Simulate an interruption** by stopping the program.
        3. **Resume the computation** from the last checkpoint.

        """
    )
    return


@app.cell
def _(load_checkpoint, save_checkpoint, time):
    def perform_computation(start, end, checkpoint_file='checkpoint.pkl'):
        """Perform a computation with checkpointing."""
        for i in range(start, end):
            # Simulate some computation
            print(f"Computing iteration {i}")
            time.sleep(0.5)  # Simulate time-consuming work
        
            # Save checkpoint every 5 iterations
            if i % 5 == 0:
                save_checkpoint(i, checkpoint_file)
    
        print("Computation completed.")

    # Load the last checkpoint if it exists
    checkpoint = load_checkpoint()

    # Start from the checkpoint or from the beginning
    start_iteration = checkpoint + 1 if checkpoint is not None else 1
    end_iteration = 20

    # Perform the computation with checkpointing
    perform_computation(start_iteration, end_iteration)
    return (perform_computation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        - **Checkpointing Mechanism**: The `save_checkpoint` function saves the current iteration number to a file. This simulates saving the state of a computation. The `load_checkpoint` function checks if a checkpoint file exists and loads the last saved state.
  
        - **Computation**: The computation is simulated by a loop that prints the current iteration and sleeps for half a second to simulate work. Every 5 iterations, the state is saved to the checkpoint file.

        - **Resumption**: If the program is interrupted and restarted, it will resume from the last saved iteration, rather than starting from the beginning.

        ### Optional Exercise:
        - **Simulate an Interruption**: After a few iterations, stop the notebook and restart it. Observe how the program resumes from the last saved checkpoint.
        - **Modify the Checkpoint Interval**: Change the checkpoint interval from every 5 iterations to every 2 or 10 iterations and see how it affects the resumption.
        - **Extend the Computation**: Increase the total number of iterations and observe how checkpointing helps in resuming the long-running computation.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part 3: Simulating an Error and Recovery

        In this section, we'll simulate an error during the computation to see how checkpointing can help recover from the failure. We'll introduce an intentional error in the computation, stop the program, and then restart it to resume from the last saved checkpoint.

        ### Steps:
        1. **Introduce an error** in the computation.
        2. **Simulate the recovery** by catching the error and re-running the program.
        3. **Observe the recovery** process as the program resumes from the last checkpoint.

        """
    )
    return


@app.cell
def _(load_checkpoint, save_checkpoint, time):
    def perform_computation_with_error(start, end, checkpoint_file='checkpoint.pkl'):
        """Perform a computation with an intentional error and checkpointing."""
        for i in range(start, end):
            print(f'Computing iteration {i}')
            time.sleep(0.5)
            if i == 10:
                raise RuntimeError(f'Error encountered at iteration {i}!')
            if i % 5 == 0:
                save_checkpoint(i, checkpoint_file)
        print('Computation completed.')
    checkpoint_1 = load_checkpoint()
    start_iteration_1 = checkpoint_1 + 1 if checkpoint_1 is not None else 1
    end_iteration_1 = 20
    try:
        perform_computation_with_error(start_iteration_1, end_iteration_1)
    except RuntimeError as e:
        print(e)
        print('An error occurred. Attempting to recover from the last checkpoint...')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        - **Error Simulation**: An intentional error is introduced at iteration 10 by raising a `RuntimeError`. This simulates an unexpected failure in the computation.
  
        - **Error Handling**: The error is caught by a `try-except` block, which prints an error message and attempts to recover by restarting the computation from the last checkpoint.

        - **Recovery Process**: After the error is caught, the program can be restarted, and it will automatically resume from the last checkpoint (iteration 10 in this case).

        ### Exercise:
        - **Run the Program**: Execute the code to see how the error is handled. Observe how the program resumes from the last checkpoint after the error.
        - **Change the Error Condition**: Modify the code to introduce the error at a different iteration and observe how the checkpointing mechanism adjusts.
        - **Expand the Recovery Mechanism**: Enhance the recovery process by automatically restarting the computation after an error without requiring manual intervention.

        """
    )
    return


@app.cell
def _(load_checkpoint, perform_computation):
    checkpoint_2 = load_checkpoint()
    start_iteration_2 = checkpoint_2 + 1 if checkpoint_2 is not None else 1
    end_iteration_2 = 20
    perform_computation(start_iteration_2, end_iteration_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Conclusion

        In this exercise, we've demonstrated how checkpointing can help recover from errors in a long-running computation. By periodically saving the state of the computation, we were able to resume from the last checkpoint after encountering an error, minimizing the loss of progress.

        This technique is crucial in HPC environments, where computations can run for extended periods, and unexpected failures can result in significant loss of work. By implementing checkpointing, we can ensure that our computations are more resilient to such failures.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6. Introduction to Performance Tuning and Analysis

        Performance tuning and analysis are crucial for maximizing the efficiency of HPC applications. This section introduces the fundamental steps involved in performance tuning, including identifying bottlenecks, applying optimizations, and verifying improvements.

        ### 6.1 Overview of Performance Tuning Steps

        The general workflow for performance tuning involves:
        1. **Profiling the code** to identify performance bottlenecks.
        2. **Applying optimizations** to the identified bottlenecks.
        3. **Reprofiling the code** to assess the impact of the optimizations.
        4. **Iterating** until performance goals are met.



        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7. Profiling and Identifying Bottlenecks

        In this section, we'll profile a computational code to identify the most time-consuming parts. Profiling is the first step in any performance tuning process.

        ### 7.1 Profiling with cProfile and line_profiler

        We'll use `cProfile` for an overall view of the code's performance and `line_profiler` for detailed line-by-line analysis.

        """
    )
    return


@app.cell
def _(cProfile, np):
    def compute_heavy_task(A, B):
        C = np.dot(A, B)
        D = np.linalg.inv(C)
        E = np.sum(D)
        return E
    A_1 = np.random.rand(1000, 1000)
    B_1 = np.random.rand(1000, 1000)
    cProfile.run('compute_heavy_task(A, B)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        The code above uses `cProfile` to profile the entire function and `line_profiler` for a detailed line-by-line breakdown. This helps in identifying which parts of the code are the most time-consuming.

        ### Exercise:

        Try modifying the `compute_heavy_task` function by adding other operations, such as matrix transposition or element-wise multiplication. Re-run the profiling tools to see how the performance characteristics change.

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
def _(np, time):
    def basic_matrix_sum(matrix):
        total = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                total = total + matrix[i, j]
        return total

    def vectorized_matrix_sum(matrix):
        return np.sum(matrix)
    matrix_1 = np.random.rand(10000, 10000)
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
def _(np, time):
    def row_major_sum(matrix):
        total = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                total = total + matrix[i, j]
        return total

    def column_major_sum(matrix):
        total = 0
        for j in range(matrix.shape[1]):
            for i in range(matrix.shape[0]):
                total = total + matrix[i, j]
        return total
    matrix_2 = np.random.rand(10000, 10000)
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
def _(np):
    from scipy.linalg import blas, lapack
    A_2 = np.random.rand(3, 3)
    B_2 = np.random.rand(3, 3)
    C = blas.dgemm(1.0, A_2, B_2)
    LU, piv, info = lapack.dgetrf(A_2)
    inv_matrix, info = lapack.dgetri(LU, piv)
    print('Matrix A:')
    print(A_2)
    print('\nMatrix B:')
    print(B_2)
    print('\nResult of BLAS matrix multiplication (A * B = C):')
    print(C)
    print('\nMatrix inversion of A using LAPACK:')
    print(inv_matrix)
    return blas, lapack


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
def _(np):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a large array on each process
    data = np.full(1000000, rank, dtype='i')

    # Write data collectively to a shared file
    fh = MPI.File.Open(comm, 'collective_output.dat', MPI.MODE_CREATE | MPI.MODE_WRONLY)
    fh.Write_at_all(rank * data.nbytes, data)
    fh.Close()  # Manually close the file

    # Reading data collectively
    collected_data = np.empty_like(data)
    fh = MPI.File.Open(comm, 'collective_output.dat', MPI.MODE_RDONLY)
    fh.Read_at_all(rank * collected_data.nbytes, collected_data)
    fh.Close()  # Manually close the file after reading

    # Print out a summary of the data to verify the read operation
    print(f"Process {rank}: First element = {collected_data[0]}, Last element = {collected_data[-1]}")
    return (MPI,)


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
        ## 12. Comprehensive Performance Analysis and Tuning

        In this section, we will perform a comprehensive performance analysis and tuning of a complex HPC application. We will use profiling tools to identify bottlenecks and optimize the application.

        ### 12.1 Case Study: Performance Tuning of a Scientific Application

        We will apply profiling, optimization, and parallel I/O techniques to a real-world scientific computation. The code will include matrix operations and parallel I/O.

        """
    )
    return


@app.cell
def _(MPI, blas, cProfile, lapack, np):
    comm_1 = MPI.COMM_WORLD
    rank_1 = comm_1.Get_rank()
    size_1 = comm_1.Get_size()
    N = 500
    A_3 = np.random.rand(N, N)
    B_3 = np.random.rand(N, N)

    def optimized_computation(A, B):
        C = blas.dgemm(1.0, A, B)
        LU, piv, info = lapack.dgetrf(C)
        inv_matrix, info = lapack.dgetri(LU, piv)
        result = np.sum(inv_matrix)
        return result
    cProfile.run('optimized_computation(A, B)')
    result = optimized_computation(A_3, B_3)
    file_handle = MPI.File.Open(comm_1, 'final_result.dat', MPI.MODE_CREATE | MPI.MODE_WRONLY)
    result_array = np.array([result], dtype='d')
    file_handle.Write_at_all(rank_1 * result_array.nbytes, result_array)
    file_handle.Close()
    print(f'Process {rank_1} completed its task and saved the result. Result sum: {result}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation:

        This case study brings together various optimization and parallelization techniques to solve a large-scale matrix problem. The code includes profiling, the use of high-performance libraries, and parallel I/O for saving the results.

        ### Optional Exercise:

        Expand the case study by adding more complex operations, such as eigenvalue computation or solving a system of linear equations. Profile and optimize these additional steps, and analyze how the performance scales with the problem size and number of processes.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Parallel Array Summation with MPI

        In this notebook, we will explore how to implement a parallel array summation using MPI in C. We'll start with a basic implementation and then introduce an optimization to improve performance.

        This example will guide you through:
        1. Writing a simple C program for parallel summation using MPI.
        2. Compiling and running the program in a Jupyter notebook.
        3. Implementing an optimization using OpenMP.
        4. Comparing the performance of the original and optimized versions.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 1: Write the Initial C Program

        We'll start by writing a simple C program that sums an array in parallel using MPI. The program initializes an array, distributes it across multiple processes, and then each process computes the sum of its portion. Finally, the root process collects the sums and computes the total.

        """
    )
    return


@app.cell
def _():
    c_program_with_file_output = """
    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int n = 100000;  // Size of the array
        int *array = NULL;
        int local_sum = 0;

        if (rank == 0) {
            array = (int*)malloc(n * sizeof(int));
            for (int i = 0; i < n; i++) {
                array[i] = 1;  // Initialize array with ones
            }
        }

        int local_n = n / size;
        int *local_array = (int*)malloc(local_n * sizeof(int));

        MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < local_n; i++) {
            local_sum += local_array[i];
        }

        int total_sum = 0;
        MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Total sum = %d\\n", total_sum);
            free(array);
        }

        free(local_array);
        MPI_Finalize();

        return 0;
    }
    """

    # Save the C program to a file
    with open("mpi_sum_file_output.c", "w") as file:
        file.write(c_program_with_file_output)

    print("C program with file output written to mpi_sum_file_output.c")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 2: Compile and Run the Initial Program

        Now that we've written our initial C program, let's compile it using `mpicc` and run it with `mpirun`. We'll use 4 processes to demonstrate the parallel execution.

        """
    )
    return


@app.cell
def _(subprocess):
    # Compile the C program
    compile_process = subprocess.run(["mpicc", "-o", "mpi_sum_file_output", "mpi_sum_file_output.c"], capture_output=True, text=True)

    # Check if compilation was successful
    if compile_process.returncode == 0:
        print("Compilation successful.")
    else:
        print("Compilation failed:")
        print(compile_process.stderr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        create the Slurm sbatch file
        """
    )
    return


@app.cell
def _():
    slurm_script = '#!/bin/bash\n#SBATCH --job-name=mpi_sum_job\n#SBATCH --output=slurm_output.txt\n#SBATCH --ntasks=2\n#SBATCH --time=00:05:00\n#SBATCH --partition=slurmpar\n\nmpirun ./mpi_sum_file_output\n'
    with open('mpi_sum_job.slurm', 'w') as file_1:
        file_1.write(slurm_script)
    print('SLURM batch script written to mpi_sum_job.slurm')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Submit the slurm job
        """
    )
    return


@app.cell
def _(subprocess, time):
    # Submit the SLURM job
    submit_process = subprocess.run(["sbatch", "mpi_sum_job.slurm"], capture_output=True, text=True)

    # Check if submission was successful
    if submit_process.returncode == 0:
        print("SLURM job submitted successfully.")
        print(submit_process.stdout)
    else:
        print("Failed to submit SLURM job:")
        print(submit_process.stderr)

    # Wait a bit for the job to complete and then read the output
    time.sleep(5)  # Adjust the sleep time depending on how long the job might take

    # Read and print the contents of the SLURM output file
    try:
        with open("slurm_output.txt", "r") as f:
            output = f.read()
        print("SLURM Job Output:")
        print(output)
    except FileNotFoundError:
        print("SLURM output file not found. The job may not have run correctly.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 3: Optimize the Program

        We'll now optimize our program using OpenMP to parallelize the summation within each MPI process. This can lead to significant performance improvements, especially on multi-core systems.

        """
    )
    return


@app.cell
def _():
    c_program_optimized = '\n#include <mpi.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <omp.h>  // Include OpenMP for parallel reduction\n\nint main(int argc, char** argv) {\n    MPI_Init(&argc, &argv);\n\n    int rank, size;\n    MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n    MPI_Comm_size(MPI_COMM_WORLD, &size);\n\n    int n = 1000000;\n    int *array = NULL;\n    int local_sum = 0;\n\n    if (rank == 0) {\n        array = (int*)malloc(n * sizeof(int));\n        for (int i = 0; i < n; i++) {\n            array[i] = 1;\n        }\n    }\n\n    int local_n = n / size;\n    int *local_array = (int*)malloc(local_n * sizeof(int));\n\n    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);\n\n    // Use OpenMP for parallel reduction to optimize the summing loop\n    #pragma omp parallel for reduction(+:local_sum)\n    for (int i = 0; i < local_n; i++) {\n        local_sum += local_array[i];\n    }\n\n    int total_sum = 0;\n    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);\n\n    if (rank == 0) {\n        printf("Total sum after optimization = %d\\n", total_sum);\n        free(array);\n    }\n\n    free(local_array);\n    MPI_Finalize();\n\n    return 0;\n}\n'
    with open('mpi_sum_optimized.c', 'w') as file_2:
        file_2.write(c_program_optimized)
    print('Optimized C program written to mpi_sum_optimized.c')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Step 4: Compile and Run the Optimized Program

        With the optimization in place, let's compile the program again, this time with OpenMP enabled, and then run it to see if there is an improvement in performance.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Compile the optimized C program
    !mpicc -fopenmp -o mpi_sum_optimized mpi_sum_optimized.c

    # Run the optimized program with 4 processes
    !mpirun --allow-run-as-root -np 4 ./mpi_sum_optimized
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Conclusion and Exercises

        In this notebook, we demonstrated a basic MPI program for parallel summation and then applied an optimization using OpenMP. 

        ### Exercises:
        1. **Increase the Array Size**: Modify the array size (`n`) and observe how the performance scales with larger arrays.
        2. **Compare Execution Time**: Measure the execution time of the original and optimized versions to quantify the performance improvement.
        3. **Parallelize Further**: Experiment with parallelizing other parts of the code or using different optimization techniques, such as loop unrolling or vectorization.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Debugging and Solving a Simple Deadlock

        ## Introduction

        A deadlock occurs when two or more processes wait indefinitely for each other to release resources, causing the program to hang. This is a common issue in parallel computing and can be difficult to detect and resolve.

        In this section, we'll simulate a simple deadlock using Python's `multiprocessing` library. We'll then debug the deadlock and modify the code to resolve it.

        ### Part 1.1: Simulating a Deadlock

        Let's start by creating a scenario where two processes each wait for a lock held by the other, causing a deadlock.

        """
    )
    return


@app.cell
def _(time):
    import multiprocessing as mp
    import threading

    def process_1(lock1, lock2):
        with lock1:
            print("Process 1 acquired lock1")
            time.sleep(1)
            with lock2:
                print("Process 1 acquired lock2")
        print("Process 1 completed")

    def process_2(lock1, lock2):
        with lock2:
            print("Process 2 acquired lock2")
            time.sleep(1)
            with lock1:
                print("Process 2 acquired lock1")
        print("Process 2 completed")

    def terminate_after_timeout(processes, timeout):
        time.sleep(timeout)
        for p in processes:
            if p.is_alive():
                print("Forcing termination due to deadlock...")
                p.terminate()
                p.join()

    if __name__ == "__main__":
        # Create two locks
        lock1 = mp.Lock()
        lock2 = mp.Lock()

        # Create and start two processes
        p1 = mp.Process(target=process_1, args=(lock1, lock2))
        p2 = mp.Process(target=process_2, args=(lock1, lock2))

        p1.start()
        p2.start()

        # Start a monitoring thread to forcefully terminate the processes after 5 seconds
        timeout_thread = threading.Thread(target=terminate_after_timeout, args=([p1, p2], 5))
        timeout_thread.start()

        # Wait for the processes to complete or be terminated
        p1.join()
        p2.join()

        print("Processes have been either completed or terminated.")
    return (mp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Part 1.2: Identifying the Deadlock

        When you run the code, you'll notice that the program hangs. This is because both processes are waiting for each other to release the locks they hold, resulting in a deadlock.

        ### Part 1.3: Resolving the Deadlock

        To resolve the deadlock, we need to ensure that both processes acquire the locks in the same order. This avoids the circular wait condition that leads to a deadlock.

        """
    )
    return


@app.cell
def _(mp, time):
    def process_1_resolved(lock1, lock2):
        with lock1:
            print('Process 1 acquired lock1')
            time.sleep(1)
            with lock2:
                print('Process 1 acquired lock2')
        print('Process 1 completed')

    def process_2_resolved(lock1, lock2):
        with lock1:
            print('Process 2 acquired lock1')
            time.sleep(1)
            with lock2:
                print('Process 2 acquired lock2')
        print('Process 2 completed')
    if __name__ == '__main__':
        lock1_1 = mp.Lock()
        lock2_1 = mp.Lock()
        p1_1 = mp.Process(target=process_1_resolved, args=(lock1_1, lock2_1))
        p2_1 = mp.Process(target=process_2_resolved, args=(lock1_1, lock2_1))
        p1_1.start()
        p2_1.start()
        p1_1.join()
        p2_1.join()
        print('Both processes completed successfully.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Part 1.4: Explanation and Conclusion

        By ensuring that both processes acquire the locks in the same order, we avoid the circular wait condition that causes the deadlock. Now, the program completes successfully without hanging.

        Deadlocks are a common problem in parallel computing, but they can often be resolved by carefully managing the order in which locks are acquired.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Debugging and Solving a Simple Race Condition

        ## Introduction

        A race condition occurs when the outcome of a program depends on the timing of uncontrollable events, such as the order in which threads or processes execute. This can lead to inconsistent or incorrect results.

        In this section, we'll simulate a simple race condition using Python's `multiprocessing` library. We'll then debug the issue and modify the code to resolve it.

        ### Part 2.1: Simulating a Race Condition

        Let's start by creating a scenario where two processes update a shared variable simultaneously, leading to a race condition.

        """
    )
    return


@app.cell
def _(mp):
    def increment(shared_var, lock):
        for _ in range(10000):
            shared_var.value = shared_var.value + 1
    shared_var = mp.Value('i', 0)
    lock = mp.Lock()
    p1_2 = mp.Process(target=increment, args=(shared_var, lock))
    p2_2 = mp.Process(target=increment, args=(shared_var, lock))
    p1_2.start()
    p2_2.start()
    p1_2.join()
    p2_2.join()
    print(f'Final value of shared_var: {shared_var.value}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Part 2.2: Identifying the Race Condition

        When you run the code, you'll notice that the final value of `shared_var` is often less than expected (i.e., not 20000). This is because the two processes are simultaneously updating the shared variable without proper synchronization, leading to a race condition.

        ### Part 2.3: Resolving the Race Condition

        To resolve the race condition, we need to use a lock to ensure that only one process can update the shared variable at a time.

        """
    )
    return


@app.cell
def _(mp):
    def increment_with_lock(shared_var, lock):
        for _ in range(10000):
            with lock:
                shared_var.value = shared_var.value + 1
    if __name__ == '__main__':
        shared_var_1 = mp.Value('i', 0)
        lock_1 = mp.Lock()
        p1_3 = mp.Process(target=increment_with_lock, args=(shared_var_1, lock_1))
        p2_3 = mp.Process(target=increment_with_lock, args=(shared_var_1, lock_1))
        p1_3.start()
        p2_3.start()
        p1_3.join()
        p2_3.join()
        print(f'Final value of shared_var (with lock): {shared_var_1.value}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Part 2.4: Explanation and Conclusion

        By using a lock, we ensure that only one process can access the shared variable at a time, thereby preventing the race condition. The final value of `shared_var` should now consistently be 20000.

        Race conditions can lead to unpredictable and incorrect behavior in parallel programs. Using synchronization mechanisms like locks is crucial to ensure that shared resources are accessed safely.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### This is the end of the practice M3.P1
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

