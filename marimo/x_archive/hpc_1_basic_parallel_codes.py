import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Overview of Parallelism

        This notebook will guide you through the concepts and applications of parallelism in computing, particularly focusing on using Python for parallel programming.

        ## Learning Objectives
        - Understand the different architectures for parallel computing.
        - Learn the benefits and challenges of parallel programming.
        - Differentiate between serial and parallel code execution.
        - Implement parallel code in Python using `multiprocessing` and `concurrent.futures`.
        - Explore case studies in parallel programming, such as matrix multiplication and sorting algorithms.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to Parallelism

        Parallelism involves the simultaneous execution of multiple tasks or processes to improve performance and efficiency. It leverages multi-core processors, distributed systems, and GPUs to divide large problems into smaller tasks that can be executed concurrently.

        ### Types of Parallel Architectures
        - **Multicore Processors**: Multiple cores within a single processor chip.
        - **Distributed Systems**: Multiple independent computers connected through a network.
        - **GPUs**: Graphics Processing Units with hundreds or thousands of cores.

        #### Exercise 1: Exploring Parallel Architectures

        **Task**: Use a command to list the CPU information on your cluster.

        ```bash
        !lscpu
        ```

        **Question 1**: How many cores does each CPU in your cluster have?
        - [ ] 2
        - [ ] 4
        - [ ] 8
        - [ ] 16
        - [ ] Check your system information
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Benefits and Challenges of Parallel Programming

        ### Benefits
        - **Increased Performance**: Faster computation times for large tasks.
        - **Scalability**: Ability to handle larger datasets by adding more processors.
        - **Efficiency**: Better resource utilization in multi-core and distributed systems.

        ### Challenges
        - **Complexity**: More complex than serial code due to task management.
        - **Synchronization Issues**: Data races and deadlocks can occur.
        - **Debugging Difficulty**: Concurrent execution makes debugging harder.

        #### Exercise 2: Analyzing Parallel Program Performance

        **Task**: Discuss the potential speedup of a parallel program for a given task.

        **Question 2**: Which of the following is a key challenge of parallel programming?
        - [ ] Simplicity
        - [ ] Complexity
        - [ ] Reduced Debugging
        - [ ] Lack of Tools
        - [ ] Improved Resource Utilization
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Serial vs. Parallel Code

        ### Key Concepts
        - **Serial Execution**: Tasks are executed sequentially, one after the other.
        - **Parallel Execution**: Multiple tasks are executed simultaneously.

        #### Example: Calculating the Sum of a List

        ### Serial Code
        ```python
        import time

        numbers = [1, 2, 3, 4, 5]
        total_sum = 0

        start_time = time.time()
        for num in numbers:
            total_sum += num
        end_time = time.time()

        print("Total Sum:", total_sum)
        print("Time taken:", end_time - start_time, "seconds")
        ```

        ### Parallel Code
        ```python
        from multiprocessing import Pool

        def add_numbers(nums):
            return sum(nums)

        if __name__ == "__main__":
            numbers = [1, 2, 3, 4, 5]
            n_processes = 2  # Number of processes to use
            chunk_size = len(numbers) // n_processes

            start_time = time.time()
            with Pool(processes=n_processes) as pool:
                result = pool.map(add_numbers, [numbers[:chunk_size], numbers[chunk_size:]])
    
            total_sum = sum(result)
            end_time = time.time()

            print("Total Sum:", total_sum)
            print("Time taken:", end_time - start_time, "seconds")
        ```

        #### Exercise 3: Serial vs. Parallel Execution

        **Task**: Implement both serial and parallel versions of the sum calculation and compare their execution times.

        **Question 3**: Which approach is faster for larger datasets?
        - [ ] Serial
        - [ ] Parallel
        - [ ] Both are the same
        - [ ] Cannot determine
        - [ ] Depends on the data size
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Performance Metrics: Speedup and Efficiency

        ### Speedup
        Measures the improvement in performance of a parallel program compared to its serial counterpart.
        #### Formula
        \[
        \text{Speedup} = \frac{\text{Execution Time of Serial Program}}{\text{Execution Time of Parallel Program}}
        \]

        ### Efficiency
        Indicates how effectively the parallel resources are utilized.
        #### Formula
        \[
        \text{Efficiency} = \frac{\text{Speedup}}{\text{Number of Processors}}
        \]

        #### Exercise 4: Calculate Speedup and Efficiency

        **Task**: Calculate the speedup and efficiency of a parallel program given execution times.

        **Question 4**: What is the efficiency if the speedup is 4 and 4 processors are used?
        - [ ] 1 (100%)
        - [ ] 0.5 (50%)
        - [ ] 2 (200%)
        - [ ] 0.25 (25%)
        - [ ] Cannot determine
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Writing Parallel Code in Python

        ### Introduction to Multiprocessing in Python
        The `multiprocessing` module allows for the creation of multiple processes to run tasks in parallel.

        #### Example: Using `multiprocessing` Module
        ```python
        from multiprocessing import Process

        def print_square(number):
            print(f"Square: {number * number}")

        if __name__ == "__main__":
            processes = [Process(target=print_square, args=(i,)) for i in range(5)]

            for p in processes:
                p.start()
            for p in processes:
                p.join()
        ```

        #### Exercise 5: Implement a Parallel Task

        **Task**: Use the `multiprocessing` module to implement a parallel task that computes squares of numbers.

        **Question 5**: Which method starts the execution of a process in Python's `multiprocessing`?
        - [ ] run()
        - [ ] begin()
        - [ ] start()
        - [ ] execute()
        - [ ] launch()
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Parallelizing a Simple Loop

        ### Example: Parallel Loop Execution
        ```python

        def compute_square(num):
            return num * num

        if __name__ == "__main__":
            numbers = list(range(10))
            with Pool(4) as pool:
                squares = pool.map(compute_square, numbers)
            print("Squares:", squares)
        ```

        #### Exercise 6: Parallelizing a Loop

        **Task**: Implement a parallel loop to compute the squares of a range of numbers.

        **Question 6**: Which function is used to apply a function to each element of a list in parallel?
        - [ ] pool.apply()
        - [ ] pool.map()
        - [ ] pool.execute()
        - [ ] pool.launch()
        - [ ] pool.run()
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using the `concurrent.futures` Module

        The `concurrent.futures` module provides a high-level interface for asynchronously executing callables.

        ### Example: Using `ProcessPoolExecutor`
        ```python
        from concurrent.futures import ProcessPoolExecutor

        def compute_square(num):
            return num * num

        if __name__ == "__main__":
            numbers = list(range(10))
            with ProcessPoolExecutor(max_workers=4) as executor:
                squares = list(executor.map(compute_square, numbers))
            print("Squares:", squares)
        ```

        #### Exercise 7: Implement Parallel Processing with `concurrent.futures`

        **Task**: Use the `ProcessPoolExecutor` to execute a parallel task.

        **Question 7**: Which class in `concurrent.futures` is used for process-based parallelism?
        - [ ] ThreadPoolExecutor
        - [ ] AsyncExecutor
        - [ ] ProcessPoolExecutor
        - [ ] ParallelExecutor
        - [ ] FutureExecutor
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Case Study: Parallel Matrix Multiplication

        ### Serial Matrix Multiplication
        ```python
        import numpy as np

        def matrix_multiply(A, B):
            return np.dot(A, B)

        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        C = matrix_multiply(A, B)
        print("Resultant Matrix C:", C)
        ```

        ### Parallel Matrix Multiplication
        ```python

        def multiply_chunk(args):
            A_chunk, B = args
            return np.dot(A_chunk, B)

        def parallel_matrix_multiply(A, B, n_processes):
            chunk_size = len(A) // n_processes
            chunks = [(A[i:i + chunk_size], B) for i in range(0, len(A), chunk_size)]
            with Pool(n_processes) as pool:
                result_chunks = pool.map(multiply_chunk, chunks)
            return np.vstack(result_chunks)

        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        C = parallel_matrix_multiply(A, B, 4)
        print("Resultant Matrix C:", C)
        ```

        #### Exercise 8: Implement Parallel Matrix Multiplication

        **Task**: Convert the serial matrix multiplication code to a parallel version.

        **Question 8**: What is the advantage of parallelizing matrix multiplication?
        - [ ] Slower computation
        - [ ] Improved readability
        - [ ] Faster computation for large matrices
        - [ ] Simpler code
        - [ ] Better precision
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Case Study: Parallel Sorting Algorithms

        ### Parallel QuickSort
        ```python

        def quicksort(arr):
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quicksort(left) + middle + quicksort(right)

        def parallel_quicksort(arr, n_processes):
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            with Pool(n_processes) as pool:
                sorted_left, sorted_right = pool.map(quicksort, [left, right])
            return sorted_left + middle + sorted_right

        if __name__ == "__main__":
            array = [33, 10, 45, 67, 2, 13, 99, 23, 12]
            sorted_array = parallel_quicksort(array, 4)
            print("Sorted Array:", sorted_array)
        ```

        #### Exercise 9: Implement Parallel QuickSort

        **Task**: Implement the parallel QuickSort algorithm and compare it with the serial version.

        **Question 9**: How does parallel QuickSort differ from the serial version?
        - [ ] Slower sorting
        - [ ] Faster sorting
        - [ ] Same performance
        - [ ] More complex code
        - [ ] Uses fewer resources
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Parallel Image Processing

        ### Using OpenCV for Parallel Image Processing
        ```python
        import cv2

        def process_image_section(args):
            image_section, filter = args
            return cv2.filter2D(image_section, -1, filter)

        def parallel_image_processing(image_path, filter, n_processes):
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            chunk_size = height // n_processes
            chunks = [image[i*chunk_size:(i+1)*chunk_size, :] for i in range(n_processes)]

            with Pool(n_processes) as pool:
                processed_chunks = pool.map(process_image_section, [(chunk, filter) for chunk in chunks])
    
            return np.vstack(processed_chunks)

        if __name__ == "__main__":
            image_path = 'large_image.jpg'
            filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_image = parallel_image_processing(image_path, filter, 4)
            cv2.imwrite('processed_image.jpg', processed_image)
        ```

        #### Exercise 10: Parallel Image Processing with OpenCV

        **Task**: Implement parallel image processing using OpenCV and analyze the performance.

        **Question 10**: What is a major benefit of parallel image processing?
        - [ ] Faster processing of large images
        - [ ] Improved image quality
        - [ ] Simplified code
        - [ ] Reduced memory usage
        - [ ] Better color accuracy
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        In this notebook, you explored various aspects of parallel programming, including the differences between serial and parallel execution, performance metrics, and practical examples using Python's `multiprocessing` and `concurrent.futures` modules. You also implemented parallel algorithms for tasks like matrix multiplication, sorting, and image processing.

        ### Answers to Quiz Questions
        1. Check your system information
        2. Complexity
        3. Parallel
        4. 1 (100%)
        5. start()
        6. pool.map()
        7. ProcessPoolExecutor
        8. Faster computation for large matrices
        9. Faster sorting
        10. Faster processing of large images

        ### Explanations
        1. Check the system information using `lscpu` to find the number of CPU cores.
        2. Complexity is a challenge due to managing tasks and synchronization.
        3. Parallel execution can be faster for larger datasets due to concurrent task execution.
        4. Efficiency is 100% when speedup equals the number of processors used.
        5. The `start()` method begins process execution in `multiprocessing`.
        6. `pool.map()` applies a function to each list element in parallel.
        7. `ProcessPoolExecutor` is used for process-based parallelism.
        8. Parallelizing matrix multiplication speeds up computation for large matrices.
        9. Parallel QuickSort sorts subarrays concurrently, speeding up the process.
        10. Parallel processing of images improves speed for large image datasets.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

