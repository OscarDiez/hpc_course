import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Module 1. Practice 2. Simple Parallel Codes
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to simple Parallel Codes

        In High-Performance Computing (HPC), efficiently utilizing computational resources to solve problems faster is crucial. Python's `multiprocessing` module provides a robust and intuitive approach for parallel processing by allowing the creation and management of subprocesses. This is especially relevant in HPC where tasks are often CPU-bound and can benefit from the parallel execution across multiple cores of a processor.

        Parallel processing with the `multiprocessing` module effectively bypasses Python’s Global Interpreter Lock (GIL), which normally prevents multiple threads from executing Python bytecodes simultaneously. By using subprocesses instead of threads, each process gets its own Python interpreter and memory space, thus overcoming the limitations imposed by the GIL.

        In this practice, we will explore how to spawn multiple processes using the `multiprocessing` module. Each process will perform a simple computation—calculating the square of a number. This basic example serves as an introduction to the capabilities of parallel processing in Python, laying the groundwork for more complex parallel computations that are common in HPC applications, such as simulations, data analysis, and matrix operations. Understanding these fundamentals is essential for leveraging the full power of HPC resources to accelerate computation-intensive tasks.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding Serial vs. Parallel Execution in Python

        In this section, we explore the differences between serial and parallel execution using Python's `multiprocessing` module. We will compare how the execution time varies when calculating squares of numbers both serially and in parallel.

        ### Serial Execution
        In serial execution, tasks are completed one after the other. This method does not utilize additional CPU cores, which can result in slower performance for CPU-bound tasks.

        ### Parallel Execution
        Parallel execution allows multiple processes to run simultaneously, leveraging multiple CPU cores. This can significantly reduce the time required to complete CPU-intensive tasks by distributing the workload across available resources.

        ## Estimating the Value of π Using Monte Carlo Simulation

        ### What is Monte Carlo Simulation?
        Monte Carlo simulation is a statistical technique that allows us to compute an approximation of a value through random sampling. This method is often used in fields such as physics, finance, and engineering to solve problems that might be deterministic in principle but complex in practice.

        ### Monte Carlo Simulation to Estimate π
        In this exercise, we use a Monte Carlo method to estimate the value of π. The principle behind this is simple: by randomly generating points within a square that encloses a quarter circle, we can estimate π based on the ratio of points that fall inside the circle to the total number of points. 

        ![Pi Monte carlo simulation](https://upload.wikimedia.org/wikipedia/commons/8/84/Pi_30K.gif)

        ### Serial vs. Parallel Execution

        #### Serial Execution
        In serial execution, points are generated one at a time, and each point's position relative to the quarter circle is calculated sequentially. This approach does not leverage additional computational resources that might be available, such as multiple CPU cores, making it slower for a large number of points.

        #### Parallel Execution
        Parallel execution divides the task among multiple processes, allowing the simultaneous generation and evaluation of points. This method can significantly speed up the computation by utilizing multiple cores, thus demonstrating the power of parallel processing in computational tasks that are both independent and identically distributed.

        ### Implementation
        We implement both serial and parallel approaches to estimate π. The parallel computation uses Python's `multiprocessing` module, which allows us to create multiple processes that can run on different cores and handle separate chunks of the task independently. The results from each process are then combined to get 


        """
    )
    return


@app.cell
def _():
    import time
    import random
    from multiprocessing import Process, Queue

    def monte_carlo_pi_part(n, queue):
        count = 0
        for _ in range(n):
            x = random.random()
            y = random.random()
            if x ** 2 + y ** 2 <= 1:
                count = count + 1
        queue.put(count)

    def serial_monte_carlo_pi(n):
        count = 0
        for _ in range(n):
            x = random.random()
            y = random.random()
            if x ** 2 + y ** 2 <= 1:
                count = count + 1
        return 4 * count / n

    def parallel_monte_carlo_pi(total_samples, num_processes):
        queue = Queue()
        processes = []
        samples_per_process = total_samples // num_processes
        start_time = time.time()
        for _ in range(num_processes):
            p = _Process(target=monte_carlo_pi_part, args=(samples_per_process, queue))
            processes.append(p)
            p.start()
        total_count = 0
        for _ in range(num_processes):
            total_count = total_count + queue.get()
        for p in processes:
            p.join()
        end_time = time.time()
        pi_estimate = 4 * total_count / total_samples
        print(f'Parallel estimate of π: {pi_estimate}')
        print(f'Parallel execution time: {end_time - start_time} seconds')
    if __name__ == '__main__':
        n_samples = 20000000
        num_processes = 4
        print('Starting serial calculation of π:')
        start_time = time.time()
        pi_estimate = serial_monte_carlo_pi(n_samples)
        end_time = time.time()
        print(f'Serial estimate of π: {pi_estimate}')
        print(f'Serial execution time: {end_time - start_time} seconds')
        print('\nStarting parallel calculation of π:')
        parallel_monte_carlo_pi(n_samples, num_processes)
    return (random,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Parallelizing a Simple Loop with `multiprocessing.Pool`

        ### Understanding `multiprocessing.Pool`
        The `multiprocessing.Pool` class is a powerful tool in Python's multiprocessing module that simplifies the process of distributing your work among multiple worker processes. This allows for parallel processing on multi-core machines which can lead to significant reductions in execution time, especially for CPU-bound tasks.

        ### How Does a Pool Work?
        A `Pool` manages a number of worker processes and distributes tasks to them. When using a `Pool`, you don’t need to manage the worker processes yourself. Instead, you just specify the number of workers, and the pool automatically handles the task distribution, execution, and collection of results.

        ### Use Case: Parallelizing Loops
        Often in programming, you encounter loops where each iteration is independent of the others. These are perfect candidates for parallel processing. By distributing iterations across multiple processes, you can complete the entire loop significantly faster than executing it serially.

        ### Example: Computing Squares
        Consider a simple task where you need to compute the square of each number in a list. Serially, this would involve processing each number one after the other. In parallel, however, we can distribute these numbers across multiple processes, each calculating the square independently, thus completing the task more quickly.

        ### Advantages of Using a Pool
        - **Efficiency**: Utilizes all available CPU cores, reducing overall processing time.
        - **Simplicity**: The API is straightforward, abstracting much of the complexity involved in process management.
        - **Flexibility**: Offers various ways to distribute tasks (e.g., `map`, `apply`, `starmap`).

        ### Practical Example
        We will demonstrate this with a Python script that uses a `Pool` to compute the squares of numbers in a list in parallel. This example will help illustrate the reduction in execution time and the effective use of system resources.

        """
    )
    return


@app.cell
def _():
    from multiprocessing import Pool

    def _compute_square(num):
        return num * num
    if __name__ == '__main__':
        _numbers = list(range(10))
        with Pool(4) as pool:
            _squares = pool.map(_compute_square, _numbers)
        print('Squares:', _squares)
    return (Pool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Parallel Execution with `concurrent.futures`

        ### Introduction to `concurrent.futures`
        The `concurrent.futures` module provides a high-level interface for asynchronously executing callables. Introduced in Python 3.2, it abstracts away many of the complexities involved in directly managing threads or processes. The module includes `ThreadPoolExecutor` and `ProcessPoolExecutor` which encapsulate thread-based and process-based parallel execution, respectively.

        ### Why Use `concurrent.futures`?
        The `concurrent.futures` module simplifies parallel execution by managing a pool of threads or processes, handling task submission, and returning futures. Futures represent the result of a computation that may not be complete yet, allowing the execution to continue without blocking.

        ### Advantages of `ProcessPoolExecutor`
        - **Ease of Use**: The API simplifies running tasks in parallel and is easy to integrate into existing code.
        - **Flexibility**: Allows specifying the number of worker processes, letting the system allocate resources efficiently.
        - **Asynchronous Execution**: Returns future objects, enabling asynchronous programming patterns and non-blocking calls.

        ### Use Case: Calculating Squares in Parallel
        A common use case for parallel processing is the independent computation of results from a list of inputs. Here, we will demonstrate using `ProcessPoolExecutor` to calculate the squares of numbers in a list. This example illustrates the ease of setup and potential speed improvements when using this method for CPU-intensive tasks.

        ### Practical Example
        Next, we will provide a Python script using `ProcessPoolExecutor` to demonstrate how straightforward and powerful this tool can be for parallelizing a simple loop.

        """
    )
    return


@app.cell
def _():
    from concurrent.futures import ProcessPoolExecutor

    def _compute_square(num):
        return num * num
    if __name__ == '__main__':
        _numbers = list(range(10))
        with ProcessPoolExecutor(max_workers=4) as executor:
            _squares = list(executor.map(_compute_square, _numbers))
        print('Squares:', _squares)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Case Study: Parallel Matrix Multiplication

        ### Significance of Matrix Operations in Scientific Computing
        Matrix operations are fundamental to many scientific computations, including physics simulations, statistical analysis, and engineering calculations. These operations, especially matrix multiplication, are computationally intensive and often constitute the bottleneck in performance for algorithms in fields such as machine learning and numerical simulation.

        ### Suitability for Parallel Processing
        Matrix multiplication can be effectively decomposed into smaller, independent computations, making it an ideal candidate for parallel processing. Since each element of the product matrix can be calculated independently of the others, parallel algorithms can distribute these calculations across multiple processors. This distribution significantly speeds up the computation as it leverages the computational power of multiple cores simultaneously.

        ### Benefits of Parallel Matrix Multiplication
        - **Speed**: Parallel processing can drastically reduce computation time, which is crucial for handling large datasets or real-time processing.
        - **Efficiency**: Utilizing multiple cores or processors allows for more efficient use of hardware resources.
        - **Scalability**: As matrix size grows, parallel processing becomes increasingly advantageous, offering better scalability compared to serial computations.

        ### Practical Implementation
        In the following section, we will explore both serial and parallel implementations of matrix multiplication using Python's built-in list data structures and the `multiprocessing` module to facilitate parallel computation.

        """
    )
    return


@app.cell
def _(Pool, random):
    def matrix_multiply(A, B):
        rows_A = len(A)
        cols_A = len(A[0])
        cols_B = len(B[0])
        result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] = result[i][j] + A[i][k] * B[k][j]
        return result

    def multiply_chunk(args):
        A_chunk, B = args
        rows_A_chunk = len(A_chunk)
        cols_A_chunk = len(A_chunk[0])
        cols_B = len(B[0])
        result_chunk = [[0 for _ in range(cols_B)] for _ in range(rows_A_chunk)]
        for i in range(rows_A_chunk):
            for j in range(cols_B):
                for k in range(cols_A_chunk):
                    result_chunk[i][j] = result_chunk[i][j] + A_chunk[i][k] * B[k][j]
        return result_chunk

    def parallel_matrix_multiply(A, B, n_processes):
        chunk_size = len(A) // n_processes
        chunks = [(A[i:i + chunk_size], B) for i in range(0, len(A), chunk_size)]
        with Pool(n_processes) as pool:
            result_chunks = pool.map(multiply_chunk, chunks)
        result = [row for chunk in result_chunks for row in chunk]
        return result

    def create_random_matrix(rows, cols):
        return [[random.random() for _ in range(cols)] for _ in range(rows)]
    A = create_random_matrix(100, 100)
    B = create_random_matrix(100, 100)
    C_serial = matrix_multiply(A, B)
    C_parallel = parallel_matrix_multiply(A, B, 4)
    print('Resultant Matrix C (Serial):', C_serial[:5][:5])
    print('Resultant Matrix C (Parallel):', C_parallel[:5][:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Advanced Topics in Parallel Programming

        ### Scalability Considerations
        Scalability in parallel programming refers to the ability of a process or system to handle a growing amount of work or its potential to be enlarged to accommodate that growth. When developing parallel applications, it's crucial to design systems that can scale efficiently as the number of processors or tasks increases. Key considerations include:
        - **Load Balancing**: Distributing work evenly across all processors to avoid scenarios where some nodes are idle while others are overloaded.
        - **Overhead Management**: Keeping the communication and synchronization overhead to a minimum as the system scales up.

        ### Understanding Deadlocks and Race Conditions
        - **Deadlocks**: A deadlock occurs when two or more processes are each waiting for the other to release a resource they need to continue execution. This situation results in a standstill where none of the processes can proceed.
        - **Race Conditions**: A race condition happens when multiple processes or threads manipulate shared data concurrently. The final value of the shared data depends on which process/thread completes last, leading to unpredictable results if not properly managed.

        ### Synchronization Issues
        Synchronization is critical in parallel programming to ensure that multiple processes or threads can operate safely when sharing resources or data. Proper synchronization can prevent race conditions and ensure data integrity. Common synchronization mechanisms include:
        - **Locks**: Allow only one thread to access a resource at a time.
        - **Semaphores**: A more flexible mechanism that uses counters to control access to one or more shared resources.

        ### Practical Example: Using Locks to Handle Race Conditions
        To demonstrate the importance of synchronization, we'll use a Python example where multiple processes increment a shared counter. Without proper synchronization, the final count could be incorrect due to race conditions. We'll use a `Lock` to ensure that only one process can increment the counter at a time.

        """
    )
    return


@app.cell
def _():
    from multiprocessing import Process, Lock, Value

    def increment(shared_value, lock):
        with lock:
            for _ in range(100):
                shared_value.value = shared_value.value + 1
    if __name__ == '__main__':
        shared_value = Value('i', 0)
        lock = Lock()
        processes = [_Process(target=increment, args=(shared_value, lock)) for _ in range(4)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print('Final Value:', shared_value.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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

