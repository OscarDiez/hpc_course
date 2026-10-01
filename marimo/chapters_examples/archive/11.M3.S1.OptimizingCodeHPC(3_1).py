import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to Code Optimization

        In high-performance computing (HPC), code optimization is critical for maximizing system performance. The goal is to make applications run faster by improving the way they utilize hardware resources, such as CPUs, memory, and interconnects. This session will guide you through optimizing code for HPC architectures, focusing on techniques that improve computational performance and memory efficiency. Understanding these concepts is essential in fields like scientific computing, where efficient code can significantly reduce computation time.

        We will cover:
        - The importance of code optimization in HPC.
        - An overview of HPC architectures and how their optimization requirements differ.
        - How algorithm efficiency relates to computational performance.

        Let's start by exploring code profiling techniques, a vital step in identifying performance bottlenecks.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Code Profiling Techniques

        Code profiling is an essential process for optimizing applications in high-performance computing (HPC). It involves measuring different aspects of a program's execution, such as CPU usage, memory access patterns, and I/O operations, to identify bottlenecks and improve performance. Profiling allows developers to pinpoint performance hotspots—specific sections of the code where most execution time is spent—and focus their optimization efforts where they will have the most significant impact.

        In the HPC context, profiling is even more critical, as the performance bottlenecks in parallel applications can be complex, involving synchronization, communication, and load balancing issues. Profiling also helps you understand how well the code scales across multiple nodes and cores, ensuring that optimizations lead to better parallel performance.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Algorithmic Choices for HPC

        In High-Performance Computing (HPC), selecting the right algorithms is critical for maximizing performance and scalability across multiple processors. Unlike traditional serial algorithms, parallel algorithms aim to divide computational tasks into smaller independent pieces that can be processed simultaneously. This ability to decompose a problem into independent tasks and distribute it across many processors can greatly reduce computation time and improve resource utilization.

        In this section, we will explore parallel algorithms, particularly focusing on sorting with parallel quicksort. We will also touch on complexity analysis in parallel computing, emphasizing how the performance of an algorithm scales with an increasing number of processors.

        ### Objectives:
        1. Understand the principles of parallel algorithms.
        2. Learn how to implement and run a parallel quicksort using MPI.
        3. Analyze the performance of parallel algorithms with respect to complexity.
   
        ### Example:
        We will implement parallel quicksort using MPI, where a dataset is divided among processors, sorted independently, and then merged. Parallel quicksort can potentially reduce sorting time from O(n log n) to O(log^2 n) with optimal parallelization and minimal communication overhead.

        """
    )
    return


@app.cell
def _():
    import os
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    return


app._unparsable_cell(
    r"""
    # Write the C code for parallel quicksort with MPI to a file
    code = \"\"\"
    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>

    // Function to swap elements in the array
    void swap(int* a, int* b) {
        int temp = *a;
        *a = *b;
        *b = temp;
    }

    // Standard quicksort partition function
    int partition(int arr[], int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[high]);
        return i + 1;
    }

    // Quicksort algorithm
    void quicksort(int arr[], int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quicksort(arr, low, pi - 1);
            quicksort(arr, pi + 1, high);
        }
    }

    // Merge function to combine sorted arrays
    void merge(int* arr1, int size1, int* arr2, int size2, int* result) {
        int i = 0, j = 0, k = 0;
        while (i < size1 && j < size2) {
            if (arr1[i] < arr2[j]) {
                result[k++] = arr1[i++];
            } else {
                result[k++] = arr2[j++];
            }
        }
        while (i < size1) result[k++] = arr1[i++];
        while (j < size2) result[k++] = arr2[j++];
    }

    int main(int argc, char** argv) {
        int size, rank;

        MPI_Init(&argc, &argv); // Initialize MPI
        MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

        int n = 16; // Number of elements in the array
        int* arr = NULL;
        int* local_arr;
        int local_n;

        // The master process initializes the array and scatters it
        if (rank == 0) {
            arr = (int*)malloc(n * sizeof(int));
            printf(\"Original array: \");
            for (int i = 0; i < n; i++) {
                arr[i] = rand() % 100; // Random numbers between 0 and 99
                printf(\"%d \", arr[i]);
            }
            printf(\"\\n\");
        }

        // Divide the array into chunks for each process
        local_n = n / size;
        local_arr = (int*)malloc(local_n * sizeof(int));

        // Scatter the array to all processes
        MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

        // Each process sorts its part
        quicksort(local_arr, 0, local_n - 1);

        // Gather the sorted parts back into the original array on the root process
        MPI_Gather(local_arr, local_n, MPI_INT, arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

        // The root process merges all sorted parts
        if (rank == 0) {
            int* sorted_arr = (int*)malloc(n * sizeof(int));

            // Initial merge of the first two parts
            merge(arr, local_n, arr + local_n, local_n, sorted_arr);

            // Now merge any remaining sorted parts (if more than two processes)
            for (int i = 2 * local_n; i < n; i += local_n) {
                int* temp_arr = (int*)malloc(n * sizeof(int));
                merge(sorted_arr, i, arr + i, local_n, temp_arr);
                for (int j = 0; j < i + local_n; j++) {
                    sorted_arr[j] = temp_arr[j];
                }
                free(temp_arr);
            }

            printf(\"Sorted array: \");
            for (int i = 0; i < n; i++) {
                printf(\"%d \", sorted_arr[i]);
            }
            printf(\"\\n\");

            free(sorted_arr);
            free(arr);
        }

        free(local_arr);
        MPI_Finalize(); // Finalize MPI
        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"parallel_quicksort.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with mpicc (MPI compiler)
    !mpicc -o parallel_quicksort parallel_quicksort.c

    # Run the compiled program with mpirun using --oversubscribe (4 processes)
    !mpirun --oversubscribe -np 4 ./parallel_quicksort
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explanation of Parallel Quicksort with MPI

        ### Code Breakdown:
        1. **MPI Initialization**: The program begins by initializing the MPI environment with `MPI_Init()`, retrieving the number of processes (`MPI_Comm_size()`) and the rank of each process (`MPI_Comm_rank()`).

        2. **Array Initialization**: On the root process (`rank == 0`), an array of 16 random integers is created. This array will be scattered to all processes for sorting.

        3. **Quicksort Algorithm**: Each process receives a chunk of the array and applies the standard quicksort algorithm locally. Quicksort is a divide-and-conquer algorithm that recursively partitions the array around a pivot element and sorts the partitions independently.

        4. **Scatter and Gather**: MPI’s `MPI_Scatter()` is used to distribute chunks of the array to each process, and `MPI_Gather()` is used to collect the sorted chunks back into the root process.

        5. **Merging Sorted Chunks**: Once each process has sorted its portion of the array, the root process merges the sorted parts into a single sorted array using the `merge()` function.

        6. **Parallel Execution**: This program runs with 4 processes (as specified by `mpirun -np 4`), allowing the sorting of different sections of the array in parallel.

        ### Benefits of Parallel Quicksort:
        Parallel quicksort leverages multiple processors to independently sort portions of the data. This reduces the overall sorting time by taking advantage of concurrency. Although sorting individually on each process introduces complexity during the merging phase, the total execution time can be significantly reduced compared to a serial quicksort for large datasets.

        By distributing the work across processors and minimizing inter-process communication during sorting, this implementation optimizes for both computation and communication, making it a good example of parallel algorithm design.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Complexity Analysis in Parallel Computing

        Complexity analysis in parallel computing extends beyond the traditional measures of algorithm complexity, such as time and space complexity. It also considers how well an algorithm scales with the number of processors, and how it balances computation with communication overhead.

        In parallel computing, an ideal algorithm would reduce the time complexity compared to its serial counterpart. For instance, a task that takes O(n) time on one processor could be reduced to O(n/p) time using `p` processors, assuming perfect division of labor and no communication delays. However, in reality, communication and synchronization between processors can affect the efficiency of parallel algorithms, and this overhead needs to be factored into the complexity analysis.

        ### Key Concepts:
        1. **Serial vs. Parallel Time Complexity**: The goal in parallel computing is to distribute tasks across multiple processors to reduce overall execution time. The effectiveness of this distribution is reflected in how the time complexity changes when moving from serial to parallel computation.
   
        2. **Computation and Communication Trade-offs**: In parallel computing, reducing the computational load by dividing the work among processors must be balanced with the communication overhead between those processors. Algorithms with low communication demands scale better with increasing numbers of processors.

        3. **Speedup and Efficiency**: Speedup refers to how much faster an algorithm runs in parallel compared to serial execution. Ideally, an algorithm running on `p` processors would be `p` times faster than its serial counterpart. Efficiency measures how well an algorithm utilizes the available processors.

        In this section, we will explore examples of common algorithms (e.g., sorting, searching, matrix multiplication) and their parallel performance in terms of time complexity.

        ### Objectives:
        1. Understand the difference between serial and parallel time complexity.
        2. Explore the trade-offs between computation and communication in parallel algorithms.
        3. Analyze the speedup and efficiency of parallel algorithms using complexity analysis.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for summing an array in parallel using MPI
    code = \"\"\"
    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>

    // Function to sum elements of an array
    double sumArray(double* array, int size) {
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += array[i];
        }
        return sum;
    }

    int main(int argc, char** argv) {
        int rank, numProcs;
        const int arraySize = 1600000;  // Large array for timing comparison
        double globalSum = 0.0;

        MPI_Init(&argc, &argv);  // Initialize MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);  // Get number of processes

        // Determine chunk size for each process
        int chunkSize = arraySize / numProcs;
        double* array = NULL;
        double* localArray = (double*)malloc(chunkSize * sizeof(double));

        // Master process initializes the array
        if (rank == 0) {
            array = (double*)malloc(arraySize * sizeof(double));
            for (int i = 0; i < arraySize; i++) {
                array[i] = (double)(i + 1);  // Fill array with values 1, 2, 3, ..., arraySize
            }
        }

        // Start timing the parallel computation
        double startTime = MPI_Wtime();

        // Scatter the array to all processes
        MPI_Scatter(array, chunkSize, MPI_DOUBLE, localArray, chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Each process computes its local sum
        double localSum = sumArray(localArray, chunkSize);

        // Reduce local sums into the global sum
        MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // End timing the parallel computation
        double endTime = MPI_Wtime();

        // The master process prints the result and timing
        if (rank == 0) {
            printf(\"Global sum: %.2f\\n\", globalSum);
            printf(\"Time taken with %d processors: %.6f seconds\\n\", numProcs, endTime - startTime);
        }

        // Clean up
        free(localArray);
        if (rank == 0) {
            free(array);
        }

        MPI_Finalize();  // Finalize MPI
        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"parallel_sum.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with mpicc (MPI compiler)
    !mpicc -o parallel_sum parallel_sum.c

    # Run the compiled program with mpirun using --oversubscribe (4 processes)
    !mpirun --oversubscribe -np 4 ./parallel_sum
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explanation of MPI Program for Parallel Sum

        ### Code Breakdown:

        1. **Array Initialization**:
           - In the master process (rank 0), a large array of size 1,600,000 is initialized with values ranging from 1 to the array size. This array is distributed across multiple processors.

        2. **Scatter Operation**:
           - The `MPI_Scatter()` function divides the array into chunks and sends each chunk to a different process. Each process receives a portion of the array and computes a local sum of its chunk.

        3. **Parallel Summation**:
           - Each process calls the `sumArray()` function to compute the sum of the elements in its chunk. These local sums are then combined using `MPI_Reduce()`, which reduces all local sums into a global sum on the master process.

        4. **Timing**:
           - The program uses `MPI_Wtime()` to measure the time taken for the parallel computation. The master process (rank 0) prints the global sum and the total time taken for the parallel sum operation.

        ### Complexity Analysis:

        - **Serial Time Complexity**:
           - In a serial implementation, summing an array of size `n` would take O(n) time.
   
        - **Parallel Time Complexity**:
           - With `p` processors, the time complexity for computation is O(n/p), since each processor sums only `n/p` elements. However, there is also communication overhead for scattering the array and reducing the local sums.
           - The total parallel time complexity is O(n/p) + O(log p), where O(log p) accounts for the communication time.

        ### Speedup and Efficiency:

        - **Speedup**:
           - Speedup measures how much faster the parallel algorithm runs compared to the serial version. Ideally, with `p` processors, we would expect a speedup of `p` (i.e., the parallel version should be `p` times faster). However, communication overhead can reduce this speedup.

        - **Efficiency**:
           - Efficiency is the ratio of the speedup to the number of processors. It indicates how well the parallel algorithm scales. An efficiency close to 1 means the algorithm scales well with the number of processors.

        ### Trade-offs Between Computation and Communication:

        - As the number of processors increases, the computational load on each processor decreases, improving computation time. However, communication overhead (for scattering and reducing the data) can become more significant, limiting the speedup and reducing the efficiency of the parallel algorithm.

        - For an optimal parallel algorithm, the communication overhead should be minimized to achieve the best possible speedup.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Time Complexity Differences

        Understanding the differences in time complexity is crucial for analyzing the performance of algorithms. Time complexity describes how the runtime of an algorithm scales as the size of the input (n) increases. Different classes of time complexity—such as O(1), O(log n), O(n), O(n log n), O(n^2), and O(2^n)—grow at different rates, which can drastically affect the performance of algorithms on large datasets.

        ### Key Concepts:
        1. **O(1)**: Constant time, where the algorithm's runtime is independent of the input size.
        2. **O(log n)**: Logarithmic time, typically seen in divide-and-conquer algorithms like binary search.
        3. **O(n)**: Linear time, where the runtime grows proportionally to the input size.
        4. **O(n log n)**: Seen in efficient sorting algorithms like merge sort and quicksort.
        5. **O(n^2)**: Quadratic time, typically resulting from nested loops, where each iteration depends on the entire input size.
        6. **O(2^n)**: Exponential time, where the runtime doubles with each additional element, as seen in recursive algorithms like naive Fibonacci calculation.

        In this section, we will implement these complexities and measure their performance for increasing input sizes.

        """
    )
    return


@app.cell
def _():
    import time
    import random
    import numpy as np
    import math

    # Function to measure time taken by a function
    def measure_time(func, *args):
        start = time.time()
        result = func(*args)
        end = time.time()
        return result, end - start

    # O(1) - Constant time operation
    def constant_time_operation():
        return 42  # This function always returns 42 in constant time.

    # O(log n) - Binary Search (logarithmic time)
    def binary_search(arr, target):
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return True
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return False

    # O(n) - Summing an array (linear time)
    def sum_array(arr):
        return sum(arr)

    # O(n log n) - Merge Sort (log-linear time)
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        return merge(left, right)

    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    # O(n^2) - Pairwise comparison (quadratic time)
    def quadratic_operation(arr):
        count = 0
        for i in range(len(arr)):
            for j in range(len(arr)):
                if arr[i] < arr[j]:
                    count += 1
        return count

    # O(2^n) - Naive recursive Fibonacci (exponential time)
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    # Test input sizes
    input_sizes = [10, 100, 1000, 10000, 100000]

    # Results for different time complexities
    for n in input_sizes:
        arr = np.random.randint(1, 100, size=n)  # Generate a random array of size n

        # O(1) - Constant time
        _, time_taken = measure_time(constant_time_operation)
        print(f"O(1) - Constant time with input size {n}: {time_taken:.6f} seconds")

        # O(log n) - Binary Search
        sorted_arr = sorted(arr)
        target = random.choice(sorted_arr)
        _, time_taken = measure_time(binary_search, sorted_arr, target)
        print(f"O(log n) - Binary search with input size {n}: {time_taken:.6f} seconds")

        # O(n) - Summing an array
        _, time_taken = measure_time(sum_array, arr)
        print(f"O(n) - Summing an array with input size {n}: {time_taken:.6f} seconds")

        # O(n log n) - Merge Sort
        _, time_taken = measure_time(merge_sort, arr)
        print(f"O(n log n) - Merge sort with input size {n}: {time_taken:.6f} seconds")

        # O(n^2) - Quadratic operation
        if n <= 1000:  # Limit n for quadratic to avoid long computation times
            _, time_taken = measure_time(quadratic_operation, arr)
            print(f"O(n^2) - Quadratic operation with input size {n}: {time_taken:.6f} seconds")

    # O(2^n) - Fibonacci (Note: Limiting n due to exponential growth)
    fibonacci_sizes = [10, 20, 30]
    for n in fibonacci_sizes:
        _, time_taken = measure_time(fibonacci, n)
        print(f"O(2^n) - Fibonacci with input size {n}: {time_taken:.6f} seconds")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explanation of Time Complexity Comparison Code

        ### Code Breakdown:

        1. **O(1) - Constant Time**:
           - The constant time function (`constant_time_operation`) always returns a constant value (42). This is an example of an operation that takes the same amount of time regardless of the input size.

        2. **O(log n) - Binary Search**:
           - Binary search divides the input array in half during each iteration, resulting in logarithmic time complexity. For each input size, we search for a random target in a sorted array and measure the time it takes.

        3. **O(n) - Summing an Array**:
           - Summing an array is a linear operation, where each element of the array is processed exactly once. The time taken increases proportionally with the input size.

        4. **O(n log n) -

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Data Structure Considerations for MPI

        When working with MPI (Message Passing Interface), it’s essential to carefully select data structures that minimize communication overhead and maximize performance across distributed processes. The choice of data structures directly influences how data is partitioned and transmitted between processes running on different cores or machines.

        ### Key Considerations:
        1. **Contiguous Data Structures**: Arrays and matrices stored in a contiguous manner (e.g., row-major or column-major order) are advantageous in MPI. Functions like `MPI_Send` and `MPI_Recv` are optimized for handling contiguous blocks of data, allowing efficient communication of large datasets with minimal overhead.
   
        2. **Non-Contiguous Data Structures**: Linked lists and other non-contiguous data structures pose challenges in MPI, as their scattered memory layout complicates the communication process. This often requires the use of advanced MPI features like derived data types or manual packing and unpacking, which introduces complexity and potential inefficiencies.

        3. **Collective Operations**: MPI provides collective operations such as `MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`, and `MPI_Allreduce` that benefit significantly from optimized data structures. Arrays and matrices, due to their contiguous memory layout, tend to work well with these operations. For instance, `MPI_Allreduce` can efficiently sum elements across multiple processes.

        In this section, we will demonstrate how contiguous arrays are handled in MPI, and how to use `MPI_Allreduce` to efficiently compute the sum of array elements across processes.

        ### Objectives:
        1. Learn how to distribute contiguous data (e.g., arrays) across processes using MPI.
        2. Implement a reduction operation (`MPI_Allreduce`) to sum array elements from different processes.
        3. Compare the efficiency of using contiguous vs. non-contiguous data structures in an MPI context.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for comparing array and linked list performance with MPI
    code = \"\"\"
    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>

    // Linked list node structure
    struct Node {
        double data;
        struct Node* next;
    };

    // Function to create a linked list of a given size
    struct Node* createLinkedList(int size) {
        struct Node* head = (struct Node*)malloc(sizeof(struct Node));
        struct Node* current = head;
        for (int i = 0; i < size; i++) {
            current->data = (double)(i + 1);
            if (i == size - 1) {
                current->next = NULL;
            } else {
                current->next = (struct Node*)malloc(sizeof(struct Node));
                current = current->next;
            }
        }
        return head;
    }

    // Function to convert a linked list to an array (packing)
    void linkedListToArray(struct Node* head, double* array, int size) {
        struct Node* current = head;
        for (int i = 0; i < size; i++) {
            array[i] = current->data;
            current = current->next;
        }
    }

    // Function to convert an array back to a linked list (unpacking)
    void arrayToLinkedList(double* array, struct Node* head, int size) {
        struct Node* current = head;
        for (int i = 0; i < size; i++) {
            current->data = array[i];
            current = current->next;
        }
    }

    // Function to sum elements of a linked list
    double sumLinkedList(struct Node* head, int size) {
        double sum = 0.0;
        struct Node* current = head;
        while (current != NULL) {
            sum += current->data;
            current = current->next;
        }
        return sum;
    }

    // Function to sum elements of an array
    double sumArray(double* array, int size) {
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += array[i];
        }
        return sum;
    }

    int main(int argc, char** argv) {
        int rank, numProcs;
        const int listSize = 160000;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

        int chunkSize = listSize / numProcs;
        double globalSum = 0.0;

        // Timers for comparison
        double arrayStart, arrayEnd, linkedListStart, linkedListEnd;

        // Array example
        double* array = (double*)malloc(listSize * sizeof(double));
        double* localArray = (double*)malloc(chunkSize * sizeof(double));

        if (rank == 0) {
            for (int i = 0; i < listSize; i++) {
                array[i] = (double)(i + 1);
            }
        }

        // Time the array-based communication and summation
        arrayStart = MPI_Wtime();
        MPI_Scatter(array, chunkSize, MPI_DOUBLE, localArray, chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        double localArraySum = sumArray(localArray, chunkSize);
        MPI_Allreduce(&localArraySum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        arrayEnd = MPI_Wtime();

        if (rank == 0) {
            printf(\"Global sum using array: %.2f\\n\", globalSum);
            printf(\"Time taken using array: %.6f seconds\\n\", arrayEnd - arrayStart);
        }

        // Linked list example
        struct Node* list = NULL;
        struct Node* localList = NULL;
        double* packedArray = (double*)malloc(chunkSize * sizeof(double));

        if (rank == 0) {
            list = createLinkedList(listSize); // Create a linked list on the root process
        }

        // Time the linked list-based communication and summation
        linkedListStart = MPI_Wtime();

        if (rank == 0) {
            linkedListToArray(list, array, listSize); // Pack linked list into array
        }

        MPI_Scatter(array, chunkSize, MPI_DOUBLE, packedArray, chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Convert the packed array back into a linked list in each process
        localList = createLinkedList(chunkSize);
        arrayToLinkedList(packedArray, localList, chunkSize);

        double localLinkedListSum = sumLinkedList(localList, chunkSize);
        MPI_Allreduce(&localLinkedListSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        linkedListEnd = MPI_Wtime();

        if (rank == 0) {
            printf(\"Global sum using linked list: %.2f\\n\", globalSum);
            printf(\"Time taken using linked list: %.6f seconds\\n\", linkedListEnd - linkedListStart);
        }

        // Clean up
        free(array);
        free(localArray);
        free(packedArray);
        MPI_Finalize();

        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"mpi_linkedlist_vs_array.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with mpicc (MPI compiler)
    !mpicc -o mpi_linkedlist_vs_array mpi_linkedlist_vs_array.c

    # Run the compiled program with mpirun using --oversubscribe (4 processes)
    !mpirun --oversubscribe -np 4 ./mpi_linkedlist_vs_array
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explanation of MPI Program: Linked List vs Array for Communication

        ### Code Breakdown:

        1. **Data Structures**:
           - We use both an **array** and a **linked list** to demonstrate the differences in performance when sending and receiving data in MPI.
           - The linked list is a non-contiguous data structure, while the array is contiguous, making communication more efficient for the array.

        2. **Array Communication**:
           - The array is directly scattered using `MPI_Scatter()` and summed using `MPI_Allreduce()`. This is the standard, efficient way to communicate data in MPI.
   
        3. **Linked List Communication**:
           - Since MPI cannot directly handle non-contiguous data structures like linked lists, we **pack** the linked list into an array before sending it.
           - After receiving the array on each process, we **unpack** it back into a linked list.
           - This packing and unpacking introduces extra overhead, making the communication of linked lists less efficient compared to arrays.

        4. **Summation**:
           - For both the array and the linked list, each process sums its portion of the data and uses `MPI_Allreduce()` to compute the global sum across all processes.
   
        5. **Timing**:
           - We use `MPI_Wtime()` to measure the time taken for the communication and summation for both the array and the linked list.
           - The program prints the global sum and the time taken for both data structures, allowing us to compare their performance.

        ### Performance Differences:

        - **Array**: Since arrays are stored contiguously in memory, MPI functions like `MPI_Scatter` and `MPI_Allreduce` can operate on them efficiently, resulting in lower communication and computation times.
        - **Linked List**: Linked lists require packing and unpacking into arrays before communication, which adds significant overhead to both communication and computation. This makes linked lists less suitable for MPI-based parallel computing.

        By comparing the times for both data structures, you will see the performance impact of using non-contiguous data structures in MPI.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Loop Optimization Techniques

        Loop optimization is a critical strategy for improving performance in HPC applications. Techniques like loop unrolling, fusion, and tiling can dramatically increase cache utilization and reduce overhead.

        We will now explore loop tiling for matrix multiplication, a common technique to enhance performance by improving data locality.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Loop Optimization Techniques in HPC

        In High-Performance Computing (HPC), loops often represent the core of computation in scientific and engineering applications. Optimizing loop structures can significantly enhance the performance of these applications. Several key loop optimization techniques are widely used to improve efficiency, reduce execution time, and optimize memory usage. This section focuses on:

        ### Key Loop Optimization Techniques:
        1. **Loop Unrolling**: Reduces the overhead of loop control and increases instruction-level parallelism (ILP) by replicating loop iterations.
        2. **Loop Fusion and Fission**: Combines (fusion) or splits (fission) loops to improve cache utilization and reduce memory access latency.
        3. **Loop Tiling**: Reorganizes loop computation into smaller blocks, optimizing cache use and reducing the number of cache misses in memory-intensive tasks like matrix operations.

        Each technique is aimed at improving specific aspects of loop execution, such as minimizing loop overhead, improving data locality, or enhancing cache performance. Let's explore these techniques in more detail with practical examples.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for loop optimization techniques to a file
    code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>

    // Example data size
    #define N 580

    // Function to initialize arrays
    void initialize_arrays(double A[], double B[], double C[], int n) {
        for (int i = 0; i < n; i++) {
            A[i] = (double)(rand() % 1000);
            B[i] = (double)(rand() % 1000);
            C[i] = 0.0;
        }
    }

    // O(n) Loop Unrolling Example
    void loop_unrolling(double A[], double B[], double C[], int n) {
        for (int i = 0; i < n; i += 4) {
            C[i] = A[i] + B[i];
            C[i + 1] = A[i + 1] + B[i + 1];
            C[i + 2] = A[i + 2] + B[i + 2];
            C[i + 3] = A[i + 3] + B[i + 3];
        }
    }

    // Loop Fusion Example
    void loop_fusion(double A[], double B[], double C[], double D[], int n) {
        for (int i = 0; i < n; i++) {
            A[i] = A[i] + B[i];
            D[i] = C[i] * A[i];
        }
    }

    // Loop Fission Example
    void loop_fission(double A[], double B[], double C[], double D[], int n) {
        for (int i = 0; i < n; i++) {
            A[i] = A[i] + B[i];
        }
        for (int i = 0; i < n; i++) {
            D[i] = C[i] * A[i];
        }
    }

    // Loop Tiling Example for Matrix Multiplication
    void matrix_multiplication_tiling(double A[N][N], double B[N][N], double C[N][N], int n, int blockSize) {
        for (int i = 0; i < n; i += blockSize) {
            for (int j = 0; j < n; j += blockSize) {
                for (int k = 0; k < n; k += blockSize) {
                    for (int ii = i; ii < i + blockSize && ii < n; ii++) {
                        for (int jj = j; jj < j + blockSize && jj < n; jj++) {
                            for (int kk = k; kk < k + blockSize && kk < n; kk++) {
                                C[ii][jj] += A[ii][kk] * B[kk][jj];
                            }
                        }
                    }
                }
            }
        }
    }

    int main() {
        // Array initialization for loop unrolling, fusion, and fission
        double A[N], B[N], C[N], D[N];
        initialize_arrays(A, B, C, N);

        // Timing loop unrolling
        clock_t start = clock();
        loop_unrolling(A, B, C, N);
        clock_t end = clock();
        printf(\"Loop Unrolling Time: %.6f seconds\\n\", (double)(end - start) / CLOCKS_PER_SEC);

        // Timing loop fusion
        start = clock();
        loop_fusion(A, B, C, D, N);
        end = clock();
        printf(\"Loop Fusion Time: %.6f seconds\\n\", (double)(end - start) / CLOCKS_PER_SEC);

        // Timing loop fission
        start = clock();
        loop_fission(A, B, C, D, N);
        end = clock();
        printf(\"Loop Fission Time: %.6f seconds\\n\", (double)(end - start) / CLOCKS_PER_SEC);

        // Matrix multiplication with loop tiling
        double M[N][N], N1[N][N], N2[N][N];
        start = clock();
        matrix_multiplication_tiling(M, N1, N2, N, 64);
        end = clock();
        printf(\"Loop Tiling (Matrix Multiplication) Time: %.6f seconds\\n\", (double)(end - start) / CLOCKS_PER_SEC);

        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"loop_optimization.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with gcc (C compiler)
    !gcc -o loop_optimization loop_optimization.c

    # Run the compiled program
    !./loop_optimization
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explanation of Loop Optimization Techniques in C

        ### Loop Unrolling:
        - **Concept**: In this example, we unroll a loop that adds elements of two arrays. By unrolling, we reduce the loop control overhead and allow multiple instructions to be executed in parallel, which can significantly improve performance on modern processors with multiple ALUs.
        - **Example**: The original loop iterated over one element at a time, but in the unrolled version, four elements are processed in each iteration.

        ### Loop Fusion:
        - **Concept**: Loop fusion combines two loops that operate over the same range into a single loop. This reduces loop overhead and improves cache locality since the data loaded in the first part of the loop can be used in the second part without being reloaded.
        - **Example**: Two loops—one adding elements of arrays `A` and `B` and another updating `D`—are fused into one loop.

        ### Loop Fission:
        - **Concept**: Loop fission, or loop splitting, breaks a single loop into multiple loops to avoid cache conflicts or reduce the working set size. This can be particularly helpful when handling large data sets that do not fit in cache.
        - **Example**: A single loop that both updates `A` and computes values for `D` is split into two separate loops. This can improve cache efficiency when large arrays are involved.

        ### Loop Tiling:
        - **Concept**: Loop tiling, also known as loop blocking, is used to break down large computations (such as matrix multiplication) into smaller blocks or tiles. This improves cache performance by ensuring that smaller portions of data are repeatedly used while they remain in the cache, reducing memory traffic between RAM and the cache.
        - **Example**: The matrix multiplication example demonstrates how loop tiling is applied to process blocks of a matrix, ensuring that the working data fits in the cache, thereby reducing cache misses.

        ### Performance Improvements:
        - **Timing Results**: The code measures the execution time of each optimization technique. Students can compare these results to understand the impact of each technique on performance.
        - **Cache and Loop Control**: By optimizing loops, we reduce unnecessary memory access, minimize cache misses, and reduce the overhead of loop control, leading to improved computational performance.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Vectorization in HPC

        Vectorization is an optimization technique that enables a single instruction to process multiple data points simultaneously using SIMD (Single Instruction, Multiple Data) capabilities. Modern CPUs include SIMD extensions like Intel's AVX and ARM's NEON, which allow operations such as addition and multiplication to be performed on multiple data elements at once. By leveraging these extensions, developers can achieve greater data parallelism, improving computational throughput.

        ### Key Aspects of Vectorization:
        1. **SIMD**: Vectorized operations execute the same instruction across multiple data points in parallel.
        2. **Data Alignment**: Ensuring that data is aligned correctly is crucial for efficient vectorization.
        3. **Compiler Directives and Intrinsics**: Compilers like GCC and Intel provide auto-vectorization and intrinsics to help optimize performance.

        ### Benefits of Vectorization:
        - Increased throughput by processing multiple data elements in parallel.
        - Enhanced performance for operations like matrix multiplication, dot products, and other numerical computations.
        - Reduced overhead and improved memory efficiency by utilizing SIMD registers to hold multiple values and process them simultaneously.

        This section will demonstrate how to apply vectorization using compiler directives, SIMD intrinsics, and practical examples of vectorized operations.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for vectorization examples using AVX but without FMA to a file
    code = \"\"\"
    #include <stdio.h>
    #include <immintrin.h>  // For AVX intrinsics
    #include <omp.h>        // For OpenMP

    // Example size
    #define N 1000000

    // Vector addition using OpenMP SIMD directive (auto-vectorization)
    void vector_add_openmp(float* x, float* y, float* z, int n, float scalar) {
        #pragma omp simd
        for (int i = 0; i < n; i++) {
            z[i] = x[i] * scalar + y[i];
        }
    }

    // Vector addition using AVX intrinsics (without FMA)
    void vector_add_intrinsics(float* x, float* y, float* z, int n, float scalar) {
        __m256 scalar_vec = _mm256_set1_ps(scalar);  // Load scalar into AVX register
        for (int i = 0; i < n; i += 8) {
            __m256 x_vec = _mm256_load_ps(&x[i]);    // Load 8 elements from x
            __m256 y_vec = _mm256_load_ps(&y[i]);    // Load 8 elements from y
            __m256 mul_vec = _mm256_mul_ps(x_vec, scalar_vec);  // Multiply x[i] * scalar
            __m256 z_vec = _mm256_add_ps(mul_vec, y_vec);       // Add the result to y[i]
            _mm256_store_ps(&z[i], z_vec);           // Store result in z
        }
    }

    // Dot product using AVX intrinsics
    double dot_product_intrinsics(const double* a, const double* b, int n) {
        __m256d sum = _mm256_setzero_pd();  // Initialize the sum
        for (int i = 0; i < n; i += 4) {
            __m256d va = _mm256_load_pd(&a[i]);  // Load 4 elements from a
            __m256d vb = _mm256_load_pd(&b[i]);  // Load 4 elements from b
            __m256d prod = _mm256_mul_pd(va, vb); // Multiply a[i] and b[i]
            sum = _mm256_add_pd(sum, prod);      // Add product to sum
        }
        double buffer[4];
        _mm256_store_pd(buffer, sum);            // Store the result
        return buffer[0] + buffer[1] + buffer[2] + buffer[3];  // Sum the buffer
    }

    int main() {
        // Allocate arrays for vector operations
        float *x = (float*)_mm_malloc(N * sizeof(float), 32);
        float *y = (float*)_mm_malloc(N * sizeof(float), 32);
        float *z = (float*)_mm_malloc(N * sizeof(float), 32);

        // Initialize arrays
        for (int i = 0; i < N; i++) {
            x[i] = (float)(i + 1);
            y[i] = (float)(i + 2);
            z[i] = 0.0f;
        }

        // Scalar for vector addition
        float scalar = 2.0f;

        // Timing vector addition with OpenMP SIMD
        double start = omp_get_wtime();
        vector_add_openmp(x, y, z, N, scalar);
        double end = omp_get_wtime();
        printf(\"OpenMP SIMD Vector Addition Time: %.6f seconds\\n\", end - start);

        // Timing vector addition with AVX intrinsics
        start = omp_get_wtime();
        vector_add_intrinsics(x, y, z, N, scalar);
        end = omp_get_wtime();
        printf(\"AVX Intrinsics Vector Addition Time: %.6f seconds\\n\", end - start);

        // Timing dot product with AVX intrinsics
        double *a = (double*)_mm_malloc(N * sizeof(double), 32);
        double *b = (double*)_mm_malloc(N * sizeof(double), 32);
        for (int i = 0; i < N; i++) {
            a[i] = (double)(i + 1);
            b[i] = (double)(i + 2);
        }

        start = omp_get_wtime();
        double dot_result = dot_product_intrinsics(a, b, N);
        end = omp_get_wtime();
        printf(\"AVX Intrinsics Dot Product Result: %.2f\\n\", dot_result);
        printf(\"AVX Intrinsics Dot Product Time: %.6f seconds\\n\", end - start);

        // Clean up
        _mm_free(x);
        _mm_free(y);
        _mm_free(z);
        _mm_free(a);
        _mm_free(b);

        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"vectorization_examples_no_fma.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with gcc (C compiler) without FMA, but with AVX
    !gcc -o vectorization_examples_no_fma vectorization_examples_no_fma.c -fopenmp -mavx -lm

    # Run the compiled program
    !./vectorization_examples_no_fma
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explanation of Vectorization Examples (Without FMA)

        ### Vector Addition Using AVX Intrinsics (Without FMA)
        - **Modified Code**: In this example, we avoid using the fused multiply-add (FMA) instruction and instead use separate AVX instructions for multiplication (`_mm256_mul_ps`) and addition (`_mm256_add_ps`).
        - **Impact**: While FMA can be more efficient by combining multiplication and addition into a single instruction, not all systems or compilers support it. By using regular AVX instructions, we maintain compatibility across more systems.

        ### Performance Comparison:
        - **OpenMP SIMD vs. AVX Intrinsics**: The program compares the performance of vector addition using OpenMP's auto-vectorization capabilities (`#pragma omp simd`) with AVX intrinsics. OpenMP provides a high-level abstraction that is easier to implement, while AVX intrinsics offer more control and potentially higher performance.
        - **Dot Product Using AVX**: The dot product example shows how AVX intrinsics can be used to process four double-precision values simultaneously. By processing multiple elements at once, the number of iterations is reduced, resulting in faster execution compared to a scalar approach.

        ### Memory Alignment:
        - **Memory Alignment**: The program uses `_mm_malloc` to allocate memory aligned to 32-byte boundaries, which is required for efficient use of AVX intrinsics. Proper alignment ensures that data is loaded efficiently into SIMD registers without penalties.

        By avoiding the use of FMA and sticking to regular AVX instructions, we ensure that the program can run on a wider range of systems while still benefiting from vectorization and SIMD parallelism.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Memory Access Patterns in HPC

        Optimizing memory access patterns is a key technique for reducing latency and increasing throughput in High-Performance Computing (HPC) applications. In many systems, memory access times can become a bottleneck due to the disparity between the speed of the processor and the speed of the memory. Understanding cache utilization and memory bandwidth is crucial to improving memory access efficiency.

        ### Key Concepts:
        1. **Cache Utilization**: Cache memory is faster than main memory but much smaller. Efficiently using the cache ensures that data needed for computation is readily available, reducing costly accesses to slower main memory. Techniques like loop tiling (as discussed previously) and **prefetching** can significantly improve cache performance.
            - **Prefetching**: This technique involves loading data into the cache before it is needed by the processor, reducing idle cycles. Prefetching can be controlled by the compiler or manually programmed.

        2. **Memory Bandwidth**: This measures the rate at which data is read from or written to memory. Maximizing memory bandwidth ensures that the processor receives data quickly and minimizes idle time. Optimizing data locality (organizing data to stay closer to the processor) and reducing unnecessary data movement are critical strategies for improving memory bandwidth.

        This section will explore techniques for improving cache utilization and memory bandwidth, including an example of prefetching and an in-place matrix transpose operation, which reduces memory bandwidth requirements by eliminating the need for extra memory allocations.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for prefetching example to a file
    code = \"\"\"
    #include <stdio.h>

    // Function to demonstrate prefetching
    void prefetching_example(float* data, int n, int prefetch_distance) {
        for (int i = 0; i < n; i++) {
            // Prefetch data before processing
            __builtin_prefetch(&data[i + prefetch_distance], 0, 1);
            // Process data
            data[i] = data[i] * 2.0f;
        }
    }

    int main() {
        int n = 100000;  // Example size
        int prefetch_distance = 64;  // Prefetch distance
        float data[n];

        // Initialize array
        for (int i = 0; i < n; i++) {
            data[i] = (float)i;
        }

        // Call prefetching example
        prefetching_example(data, n, prefetch_distance);

        // Print some of the processed results
        for (int i = 0; i < 10; i++) {
            printf(\"%f \", data[i]);
        }
        printf(\"\\n\");

        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"prefetching_example.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with gcc
    !gcc -o prefetching_example prefetching_example.c

    # Run the compiled program
    !./prefetching_example
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In-Place Matrix Transpose to Optimize Memory Bandwidth
        This example demonstrates an in-place matrix transpose operation, which improves memory locality and reduces unnecessary data movement. By avoiding extra allocations and memory copying, we conserve memory bandwidth.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for in-place matrix transpose to a file
    code = \"\"\"
    #include <stdio.h>
    #define N 4  // Size of the matrix (NxN)

    // Function to transpose matrix in place
    void transpose_inplace(float matrix[N][N]) {
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                // Swap matrix[i][j] with matrix[j][i]
                float temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }

    int main() {
        // Initialize a 4x4 matrix
        float matrix[N][N] = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        };

        // Transpose the matrix in place
        transpose_inplace(matrix);

        // Print the transposed matrix
        printf(\"Transposed Matrix:\\n\");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf(\"%f \", matrix[i][j]);
            }
            printf(\"\\n\");
        }

        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"transpose_inplace.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with gcc
    !gcc -o transpose_inplace transpose_inplace.c

    # Run the compiled program
    !./transpose_inplace
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #####Prefetching Example
        Prefetching: In this example, the __builtin_prefetch function is used to load data into the cache before it is needed. The prefetch distance (prefetch_distance) is set to 64, which means the data located 64 elements ahead of the current index is loaded into the cache in advance.
        Benefit: This technique reduces idle time by ensuring that the required data is already in the cache when the processor needs it.
        Use Case: Prefetching is especially useful in scenarios where memory access is a significant bottleneck, as it helps overlap data loading with computation.


        ####In-Place Matrix Transpose Example
        In-Place Transpose: The transpose_inplace function transposes a square matrix without creating a new matrix. By swapping elements in place, we avoid the need for additional memory allocation and reduce unnecessary data movement.
        Benefit: This approach conserves memory bandwidth by limiting data copying and improving cache locality. When accessing adjacent rows and columns, the data is more likely to stay within the cache, reducing memory latency.
        Use Case: In large-scale HPC systems, reducing data movement across memory is essential for improving performance and conserving bandwidth. In-place algorithms are particularly valuable in applications where memory efficiency is critical.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Advanced Optimization Techniques in HPC

        In High-Performance Computing (HPC), achieving peak performance requires advanced optimization techniques. These methods go beyond basic optimizations, offering ways to further enhance system throughput, reduce latency, and improve computational efficiency. This section covers three key techniques:

        ### 1. **Speculative Execution**:
        This method allows modern processors to predict the path of future conditional operations and execute tasks before they are officially required. By doing so, speculative execution keeps the CPU busy during memory access or branch resolution, which mitigates latency. The technique involves risks such as incorrect predictions, which require rolling back instructions, and the potential for security vulnerabilities like the Spectre and Meltdown flaws.

        ### 2. **Dynamic Scheduling**:
        Dynamic scheduling adjusts the assignment and order of tasks based on the system's current state. This real-time optimization is particularly useful for balancing loads across processors, ensuring all threads or cores are efficiently utilized, especially in systems where workloads vary unpredictably.

        ### 3. **Software Prefetching**:
        This method involves explicitly loading data into the cache before it is required by computation. It minimizes cache misses, which are a significant source of latency in large-scale applications. By fetching data in advance, software prefetching reduces idle processor cycles caused by slow memory access.

        These advanced techniques, when used appropriately, can lead to substantial performance gains in HPC environments, allowing for more efficient computation and reduced latency.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ####Code Example 1: Speculative Execution
        This example demonstrates speculative execution using the likely() macro, which provides hints to the compiler about the expected outcome of a condition.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for speculative execution using likely() to a file
    code = \"\"\"
    #include <stdio.h>

    #define likely(x) __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)

    // Example function using speculative execution
    void speculative_example(int x) {
        if (likely(x > 0)) {
            printf(\"x is positive\\n\");
        } else {
            printf(\"x is non-positive\\n\");
        }
    }

    int main() {
        int x = 5;
        speculative_example(x);

        x = -1;
        speculative_example(x);

        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"speculative_execution.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with gcc
    !gcc -o speculative_execution speculative_execution.c

    # Run the compiled program
    !./speculative_execution
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Dynamic Scheduling with OpenMP
        This example shows how to use dynamic scheduling in OpenMP, where tasks are dynamically assigned to threads based on their availability to optimize load balancing.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for dynamic scheduling using OpenMP to a file
    code = \"\"\"
    #include <stdio.h>
    #include <omp.h>

    // Function to simulate work
    void process(int i) {
        // Simulate some computation
        printf(\"Processing element %d by thread %d\\n\", i, omp_get_thread_num());
    }

    int main() {
        int n = 16;  // Example size
        int data[n];

        // Initialize data array
        for (int i = 0; i < n; i++) {
            data[i] = i;
        }

        // Parallel loop with dynamic scheduling
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            process(data[i]);
        }

        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"dynamic_scheduling.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with gcc and OpenMP
    !gcc -fopenmp -o dynamic_scheduling dynamic_scheduling.c

    # Run the compiled program
    !./dynamic_scheduling
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
         #### Software Prefetching
        This example demonstrates how software prefetching can be used to load data into the cache ahead of time, reducing cache misses and improving performance.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for software prefetching to a file
    code = \"\"\"
    #include <stdio.h>

    // Function to simulate work with software prefetching
    void process(float* data, int n, int prefetch_distance) {
        for (int i = 0; i < n; i++) {
            // Prefetch the next set of data
            __builtin_prefetch(&data[i + prefetch_distance], 0, 1);
            // Simulate processing
            data[i] = data[i] * 2.0f;
        }
    }

    int main() {
        int n = 100000;  // Example size
        int prefetch_distance = 64;  // Distance to prefetch
        float data[n];

        // Initialize data
        for (int i = 0; i < n; i++) {
            data[i] = (float)i;
        }

        // Process the data with software prefetching
        process(data, n, prefetch_distance);

        // Print some results to verify
        for (int i = 0; i < 10; i++) {
            printf(\"%f \", data[i]);
        }
        printf(\"\\n\");

        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"software_prefetching.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with gcc
    !gcc -o software_prefetching software_prefetching.c

    # Run the compiled program
    !./software_prefetching
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###Speculative Execution
        Concept: The likely() macro provides hints to the compiler about which branch of a condition is more likely to be taken. This allows the CPU to speculate on the most probable path of execution. In the code, when x > 0 is likely true, the CPU speculatively executes the corresponding instructions to minimize latency.
        Benefits: By predicting the branch, the CPU can execute instructions ahead of time, leading to higher throughput if the prediction is correct. However, if the prediction is wrong, the speculative instructions are discarded, which incurs some overhead.

        ###Dynamic Scheduling
        Concept: The #pragma omp parallel for schedule(dynamic) directive in OpenMP dynamically assigns tasks to threads as they become available. This helps balance the workload among threads, especially when the amount of work per task is unpredictable.
        Benefits: Dynamic scheduling ensures that no thread is left idle while others have tasks to complete. This real-time adjustment of task assignment improves resource utilization in HPC environments with variable workloads.
        Software Prefetching
        Concept: Software prefetching proactively loads data into the cache before it is needed by the processor. The __builtin_prefetch() function is used to instruct the compiler to fetch data prefetch_distance iterations ahead of the current index.
        Benefits: By reducing cache misses, software prefetching minimizes memory access latency and improves overall performance, especially in loops where data is accessed sequentially over large arrays.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Case Study: Optimization of a Matrix Multiplication Algorithm

        Matrix multiplication is a core operation in many scientific computing and machine learning applications. As a fundamental building block for various algorithms, optimizing matrix multiplication can lead to substantial performance improvements, especially in high-performance computing (HPC) environments.

        ### Basic Matrix Multiplication Algorithm:
        The standard algorithm for multiplying two n×n matrices, A and B, involves three nested loops that calculate the result matrix C:

        ```c
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        """
    )
    return


app._unparsable_cell(
    r"""
    # Write the C code for optimized matrix multiplication without FMA to a file
    code = \"\"\"
    #include <immintrin.h> // For AVX intrinsics
    #include <omp.h>       // For OpenMP
    #include <stdio.h>
    #include <stdlib.h>    // For random number generation

    // Optimized matrix multiplication with loop tiling, vectorization, dynamic scheduling, and prefetching
    void optimized_matrix_multiply(float* A, float* B, float* C, int n) {
        int blockSize = 64; // Block size for loop tiling

        // Parallelize the outer loops with dynamic scheduling for load balancing
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int ii = 0; ii < n; ii += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                for (int kk = 0; kk < n; kk += blockSize) {
                    for (int i = ii; i < ii + blockSize && i < n; i++) {
                        for (int j = jj; j < jj + blockSize && j < n; j++) {
                            __m256 sum = _mm256_setzero_ps();  // Initialize sum vector

                            for (int k = kk; k < kk + blockSize && k < n; k += 8) {
                                // Prefetch next data to minimize cache misses
                                if (k + 16 < n) {
                                    __builtin_prefetch(&A[i * n + k + 16], 0, 1);
                                    __builtin_prefetch(&B[k * n + j + 16], 0, 1);
                                }

                                // Load 8 elements from matrices A and B
                                __m256 a = _mm256_loadu_ps(&A[i * n + k]);
                                __m256 b = _mm256_loadu_ps(&B[k * n + j]);

                                // Perform multiplication and addition separately
                                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
                            }

                            // Store the result of the vector sum in C[i][j]
                            float buffer[8];
                            _mm256_storeu_ps(buffer, sum);
                            float total_sum = 0.0f;
                            for (int s = 0; s < 8; s++) {
                                total_sum += buffer[s];
                            }

                            #pragma omp atomic
                            C[i * n + j] += total_sum;  // Accumulate result into C
                        }
                    }
                }
            }
        }
    }

    int main() {
        int n = 512;  // Matrix size (512x512)
        float *A = (float*)_mm_malloc(n * n * sizeof(float), 32);
        float *B = (float*)_mm_malloc(n * n * sizeof(float), 32);
        float *C = (float*)_mm_malloc(n * n * sizeof(float), 32);

        // Initialize matrices A and B
        for (int i = 0; i < n * n; i++) {
            A[i] = (float)(rand() % 100);
            B[i] = (float)(rand() % 100);
            C[i] = 0.0f;
        }

        // Perform optimized matrix multiplication
        optimized_matrix_multiply(A, B, C, n);

        // Print part of the result matrix
        printf(\"Result Matrix (Partial):\\n\");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                printf(\"%f \", C[i * n + j]);
            }
            printf(\"\\n\");
        }

        // Free allocated memory
        _mm_free(A);
        _mm_free(B);
        _mm_free(C);

        return 0;
    }
    \"\"\"

    # Save the C code to a file
    with open(\"optimized_matrix_multiply.c\", \"w\") as file:
        file.write(code)

    # Compile the C program with gcc and OpenMP support
    !gcc -fopenmp -o optimized_matrix_multiply optimized_matrix_multiply.c -mavx -lm

    # Run the compiled program
    !./optimized_matrix_multiply
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explanation of Optimized Matrix Multiplication Code

        ### 1. **Loop Unrolling**:
        By unrolling the inner loop in chunks of 8 elements, we reduce the overhead of loop control and increase instruction-level parallelism (ILP). This allows the CPU to process more data with fewer loop iterations, improving performance.

        ### 2. **Loop Tiling (Blocking)**:
        Loop tiling breaks the matrix into smaller blocks (64x64 in this case) that fit into the cache. By operating on smaller blocks, the algorithm minimizes cache misses, improving data locality and speeding up memory access.

        ### 3. **Vectorization with AVX**:
        AVX (Advanced Vector Extensions) is used to load and process multiple elements simultaneously. In the code, the `_mm256_loadu_ps()` intrinsic loads 8 elements of matrices A and B into 256-bit registers. The fused multiply-add operation (`_mm256_fmadd_ps()`) multiplies and accumulates the results into the sum vector, reducing the number of iterations required.

        ### 4. **Dynamic Scheduling**:
        The `#pragma omp parallel for` directive with the `schedule(dynamic)` clause ensures that the workload is balanced across multiple threads dynamically. This prevents load imbalance, where some threads finish early while others are still working.

        ### 5. **Software Prefetching**:
        The `__builtin_prefetch()` function is used to prefetch data into the cache before it is needed. This reduces the chance of cache misses by fetching data into the cache in advance, ensuring that the CPU has data ready when it reaches the next iteration.

        By combining these optimization techniques—loop unrolling, loop tiling, vectorization, dynamic scheduling, and software prefetching—the algorithm achieves significant improvements in performance. These optimizations are critical in HPC applications that rely on matrix operations, such as machine learning models, numerical simulations, and data processing tasks.

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

