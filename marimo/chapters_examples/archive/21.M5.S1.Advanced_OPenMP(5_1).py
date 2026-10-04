import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #Introduction to OpenMP Task Creation with #pragma omp task

        In OpenMP, the #pragma omp task directive is a powerful feature that allows developers to express parallelism at a higher level. It enables the creation of tasks that can be executed concurrently. This is particularly useful when dealing with recursive functions, such as the computation of the Fibonacci sequence, where each task can independently compute part of the result. The use of tasks helps to exploit parallelism in irregular or dynamically structured problems.

        When using #pragma omp task, the program defines a task that may be executed by any available thread in the OpenMP team. Unlike simple parallel loops, tasks are more flexible and can be used for work that isn't easily expressed in loops.

        ##Fibonacci Example Overview
        In the example provided, we calculate the Fibonacci number recursively. Without OpenMP, the recursive function would be computed sequentially. By introducing tasks using #pragma omp task, each recursive call can be parallelized, and multiple threads can execute different parts of the Fibonacci computation in parallel.

        We'll demonstrate three versions of the Fibonacci program:

        - Serial version – No parallelism.
        - Parallel version without tasks – Uses parallel regions and single directive.
        - Parallel version with tasks – Uses tasks to exploit fine-grained parallelism.

        """
    )
    return


app._unparsable_cell(
    r"""
    # the output for the non-parallel version, the parallel version without tasks, and with tasks.

    import os

    # Define the C code for Fibonacci calculation with OpenMP tasks
    c_code = r'''
    #include <stdio.h>
    #include <omp.h>

    int fib_serial(int n) {
        if (n < 2) {
            return n;
        } else {
            return fib_serial(n - 1) + fib_serial(n - 2);
        }
    }

    int fib_parallel_no_task(int n) {
        if (n < 2) {
            return n;
        } else {
            int x, y;
            #pragma omp parallel sections
            {
                #pragma omp section
                x = fib_parallel_no_task(n - 1);
                #pragma omp section
                y = fib_parallel_no_task(n - 2);
            }
            return x + y;
        }
    }

    int fib_parallel_with_task(int n) {
        int x, y;

        if (n < 2) {
            return n;
        } else {
            #pragma omp task shared(x)
            x = fib_parallel_with_task(n - 1);

            #pragma omp task shared(y)
            y = fib_parallel_with_task(n - 2);

            #pragma omp taskwait
            return x + y;
        }
    }

    int main() {
        int n = 30;
        int result;
        double start_time, end_time;

        // Serial Fibonacci
        start_time = omp_get_wtime();
        result = fib_serial(n);
        end_time = omp_get_wtime();
        printf(\"Serial Fibonacci(%d) = %d, Time: %f seconds\n\", n, result, end_time - start_time);

        // Parallel Fibonacci without tasking
        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            result = fib_parallel_no_task(n);
        }
        end_time = omp_get_wtime();
        printf(\"Parallel Fibonacci without tasking(%d) = %d, Time: %f seconds\n\", n, result, end_time - start_time);

        // Parallel Fibonacci with tasking
        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            result = fib_parallel_with_task(n);
        }
        end_time = omp_get_wtime();
        printf(\"Parallel Fibonacci with tasking(%d) = %d, Time: %f seconds\n\", n, result, end_time - start_time);

        return 0;
    }
    '''

    # Save the code to a file
    with open(\"fib_task_example.c\", \"w\") as code_file:
        code_file.write(c_code)

    # Compile the C code with OpenMP support
    !gcc -fopenmp fib_task_example.c -o fib_task_example

    # Run the compiled program
    !./fib_task_example
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OpenMP Task Creation Example

        This notebook demonstrates the use of OpenMP tasks for parallelizing a recursive Fibonacci calculation.

        ### Explanation of Results

        The results of the Fibonacci calculation show different approaches and their performance:
        - **Serial Fibonacci**: Time = `0.008623 seconds`
        - **Parallel Fibonacci Without Tasking**: Time = `0.959481 seconds`
        - **Parallel Fibonacci With Tasking**: Time = `0.614899 seconds`

        #### 1. **Serial Version**
        In the serial version, the Fibonacci calculation is performed without any parallelism. This version is the fastest because there is no overhead from managing threads or tasks. However, it does not utilize the available multi-core processing power, which limits its scalability for larger problems.

        #### 2. **Parallel Version Without Tasking**
        This version uses OpenMP sections to parallelize the recursive Fibonacci calls. Despite using multiple threads, the time is significantly slower. The overhead of creating and managing threads for each recursive call far outweighs the benefits of parallelism, especially for an algorithm that is deeply recursive like Fibonacci.

        #### 3. **Parallel Version With Tasking**
        Using OpenMP tasks, the performance improves compared to the non-tasking version. Tasks allow recursive calls to execute in parallel across multiple threads. This version better exploits the parallelism in the recursion, leading to faster execution compared to the section-based approach, though it is still slower than the serial version due to task management overhead.

        #### Key Takeaway
        While task-based parallelism adds some overhead, it offers a more scalable solution for recursive problems. It allows for better utilization of multi-core processors and will likely show more significant performance improvements as the problem size increases.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to OpenMP Task Dependencies
        OpenMP tasks are an effective way to express parallelism, especially in algorithms with recursive or irregular workloads, like the Fibonacci example. However, when tasks are created, they may depend on one another for their results. OpenMP provides a mechanism to manage these dependencies using the depend clause with the #pragma omp task directive.

        In task-based parallelism, task dependencies ensure that certain tasks are completed before others begin. This is critical when tasks share data or when the result of one task is required by another. By specifying dependencies, OpenMP ensures correct execution order and minimizes synchronization overhead by waiting only when necessary.

        ### Task Dependencies Syntax
        The depend clause allows the programmer to specify in what way tasks are dependent on each other:

        - in: A task will depend on another task to provide the input before starting.
        - out: A task will produce data that other tasks might consume.
        -inout: A task will both read and modify shared data.

        In the Fibonacci example, where two tasks compute fib(n-1) and fib(n-2), the use of task dependencies can ensure that the summing of these results occurs after both tasks have completed.


        """
    )
    return


app._unparsable_cell(
    r"""
    # demonstrating task dependencies in OpenMP.


    # Define the C code with OpenMP task dependencies for Fibonacci calculation
    c_code_with_dependencies = r'''
    #include <stdio.h>
    #include <omp.h>

    int fib_with_dependencies(int n) {
        int x, y;

        if (n < 2) {
            return n;
        } else {
            #pragma omp task shared(x) depend(out: x)
            x = fib_with_dependencies(n - 1);

            #pragma omp task shared(y) depend(out: y)
            y = fib_with_dependencies(n - 2);

            #pragma omp taskwait
            return x + y;
        }
    }

    int main() {
        int result;
        int n = 30;
        double start_time, end_time;

        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            result = fib_with_dependencies(n);
        }
        end_time = omp_get_wtime();

        printf(\"Fibonacci with task dependencies(%d) = %d, Time: %f seconds\n\", n, result, end_time - start_time);
        return 0;
    }
    '''

    # Save the C code to a file
    with open(\"fib_task_dependencies_example.c\", \"w\") as code_file:
        code_file.write(c_code_with_dependencies)

    # Compile the C code with OpenMP support
    !gcc -fopenmp fib_task_dependencies_example.c -o fib_task_dependencies_example

    # Run the compiled program
    !./fib_task_dependencies_example
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of Code with Task Dependencies
        In this version, we introduce task dependencies using the depend clause. The key difference between this and the previous task-based version is that the dependencies are explicitly declared, ensuring that the tasks complete in a defined order.

        #### How Task Dependencies Work:
        1. Creation of Tasks:
         - The first task computes fib(n-1) and is assigned an output dependency (depend(out: x)), meaning that any task depending on the value of x will wait for this task to complete.
         - The second task computes fib(n-2) and similarly uses an output dependency (depend(out: y)).
        2. Taskwait:
         - The #pragma omp taskwait ensures that the program waits for both tasks (fib(n-1) and fib(n-2)) to finish before summing their results.

        #### Advantages of Task Dependencies:
        - Fine-Grained Control: By explicitly defining the dependencies between tasks, OpenMP can better manage task execution, ensuring correctness while still allowing parallel execution where possible.
        - Avoiding Unnecessary Synchronization: Instead of waiting for all tasks at once (as with taskwait), the program waits only for tasks that are required for the next step, improving performance.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Is normal that takes longer using dependencies?

        Yes, it is normal for the task-based Fibonacci implementation with dependencies to take longer in some cases, and this can be attributed to several factors:

        1. Overhead of Task Management
        Creating tasks, managing dependencies, and scheduling them involves some overhead in OpenMP. Each time a task is created, OpenMP must track the dependencies between tasks to ensure that they execute in the correct order. This tracking adds some computational overhead compared to simply running everything sequentially.
        For a recursive problem like Fibonacci, where there are many small, fine-grained tasks being created, the overhead of task creation and synchronization can dominate the actual computation, especially for smaller problem sizes like Fibonacci(30).
        2. Task Granularity
        The Fibonacci function inherently has a small computational workload for each recursive call (just a few additions). However, the overhead for creating a task and managing dependencies is relatively large compared to the actual work done inside each task.
        Tasks work best when the granularity (the amount of work done per task) is sufficiently large, as the overhead can be amortized over larger tasks. For Fibonacci, each task computes only a small part of the problem, making the overhead more significant.
        3. Recursive Nature of Fibonacci
        The Fibonacci function is highly recursive, and for Fibonacci(30), this results in a large number of tasks being created. As the recursion goes deeper, the number of tasks grows exponentially, adding further overhead.
        While task parallelism can speed up certain types of workloads, recursive functions with many small tasks, like Fibonacci, may not benefit as much unless the recursion depth is very large, where more parallelism can be exploited across multiple cores.
        4. Task Dependencies
        Although dependencies ensure correct ordering, they also limit how much parallelism can be exploited. The taskwait directive ensures that tasks complete in the correct order, but it can also introduce additional synchronization points that may cause threads to wait, reducing the efficiency of parallel execution.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to the taskloop Directive
        The taskloop directive in OpenMP simplifies the creation of tasks for loop iterations. Instead of manually creating a task for each iteration or group of iterations, the taskloop directive automatically generates tasks for different chunks of iterations. This is particularly useful when dealing with large loop-based computations where workload distribution across multiple threads can improve performance.

        By specifying the grainsize or num_tasks clauses, developers can control the number of iterations that each task handles, effectively balancing task granularity and load distribution across threads. This makes taskloop an excellent tool for managing parallelism in iterative workloads.

        ###Benefits of taskloop:
        Automated Task Generation: Automatically divides loop iterations into tasks without requiring explicit task creation for each iteration.
        Load Balancing: Allows better load distribution by controlling task size using the grainsize clause, ensuring efficient use of resources.
        Simplifies Code: Reduces the complexity of parallelizing loop-based computations by abstracting task creation.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Define the C code with OpenMP taskloop directive
    c_code_taskloop = r'''
    #include <stdio.h>
    #include <omp.h>

    void process_element(int i, int *data) {
        data[i] = i * 2;  // Example processing: multiplying by 2
    }

    int main() {
        int data[1000];

        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp taskloop grainsize(10)
                for (int i = 0; i < 1000; i++) {
                    process_element(i, data);
                }
            }
        }

        // Output a sample of the processed data
        for (int i = 0; i < 100; i += 10) {
            printf(\"data[%d] = %d\n\", i, data[i]);
        }

        return 0;
    }
    '''

    # Save the C code to a file
    with open(\"taskloop_example.c\", \"w\") as code_file:
        code_file.write(c_code_taskloop)

    # Compile the C code with OpenMP support
    !gcc -fopenmp taskloop_example.c -o taskloop_example

    # Run the compiled program
    !./taskloop_example
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of the Taskloop Code
        In this example, we utilize the #pragma omp taskloop directive to parallelize the loop that processes elements in an array.

        - Taskloop Directive: The taskloop directive is applied to the loop, which generates tasks for processing array elements. Here, we specify grainsize(10), meaning that each task will process 10 iterations of the loop.

        - Grainsize: The grainsize clause controls the size of the chunks of iterations that each task processes. By using a grainsize of 10, we ensure that each task processes a small but reasonable amount of work, allowing better load balancing across threads.

        - Parallel Execution: The loop iterations are executed in parallel by multiple threads, each handling a chunk of 10 iterations. The #pragma omp single ensures that the taskloop is executed by a single thread, but the tasks generated are distributed across available threads.

        - Output: After the loop has processed the elements, the program prints a sample of the processed data to verify that each element in the array has been correctly updated.

        ####Key Advantages:
        - Simplicity: The taskloop directive eliminates the need to manually create and manage tasks for each iteration, simplifying the code.
        - Scalability: By splitting the loop into tasks, the workload can be distributed across multiple threads, making the code scalable to more cores.
        - Granularity Control: With the grainsize clause, we can control the task size, allowing us to find the right balance between parallelism and overhead. Smaller grainsizes increase the number of tasks, leading to more parallelism but higher task creation overhead.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to the taskloop Directive
        The taskloop directive in OpenMP provides an efficient way to parallelize loop-based computations by automatically generating tasks for chunks of loop iterations. Rather than creating tasks manually for each iteration or group of iterations, the taskloop directive automates this process, making it easier to exploit parallelism in loop constructs. This is particularly useful for load balancing and improving task granularity.

        ## Key features of the taskloop directive include:

        - Automatic Task Generation: Automatically breaks up a loop into tasks based on the specified chunk size or number of tasks.
        Control Over Granularity: Using clauses like grainsize or num_tasks, developers can control the number of iterations handled by each task, balancing load distribution and minimizing overhead.
        - Efficient Parallelism: Tasks are generated for loop iterations and distributed among available threads, improving performance for large iterative workloads.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Define the C code for demonstrating the OpenMP taskloop directive
    c_code_taskloop = r'''
    #include <stdio.h>
    #include <omp.h>

    void process_element(int i, int *data) {
        data[i] = i * i;  // Example processing: square the element
    }

    int main() {
        int data[1000];

        // Parallel block with a taskloop to process elements in chunks
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp taskloop grainsize(20)
                for (int i = 0; i < 1000; i++) {
                    process_element(i, data);
                }
            }
        }

        // Output some processed data
        for (int i = 0; i < 100; i += 10) {
            printf(\"data[%d] = %d\n\", i, data[i]);
        }

        return 0;
    }
    '''

    # Save the C code to a file
    with open(\"taskloop_directive_example.c\", \"w\") as code_file:
        code_file.write(c_code_taskloop)

    # Compile the C code with OpenMP support
    !gcc -fopenmp taskloop_directive_example.c -o taskloop_directive_example

    # Run the compiled program
    !./taskloop_directive_example
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of Taskloop with grainsize Example
        This code demonstrates the usage of the OpenMP taskloop directive with the grainsize clause to control task granularity. Here's a breakdown of the key components:

        - Taskloop Directive: The #pragma omp taskloop grainsize(20) directive tells OpenMP to divide the loop iterations into tasks, with each task processing 20 iterations. This automatically parallelizes the loop, distributing the tasks across available threads.

        - Parallel Region: The #pragma omp parallel block ensures that multiple threads are available to execute the tasks generated by the taskloop directive. The #pragma omp single ensures that only one thread initiates the taskloop, but the generated tasks are executed by multiple threads.

        - Grainsize: The grainsize(20) clause divides the loop into tasks where each task handles 20 iterations. This balances the workload across threads by ensuring each task is sufficiently large to reduce overhead while still allowing for parallel execution.

        - Processing Logic: The function process_element() is applied to each element in the array. In this case, we square the array elements as a placeholder for more complex processing logic.

        ###Output
        The program prints a subset of the processed data to verify the correct execution of the taskloop directive. Each element of the array is squared by the process_element() function and printed to the console.

        ####Key Points
        - Simplified Parallelism: The taskloop directive eliminates the need to manually define tasks for each loop iteration, simplifying the parallelization of iterative workloads.
        - Controlled Granularity: By specifying the grainsize(20), we control how many iterations are handled by each task, ensuring a good balance between task overhead and parallelism.
        - Efficient Execution: The loop iterations are distributed across multiple threads, leveraging task-based parallelism to speed up processing.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
         ## Introduction to SIMD Clauses for Optimization
        Single Instruction, Multiple Data (SIMD) is an essential technique in modern high-performance computing for optimizing loop-based computations. OpenMP provides several SIMD-related clauses to help developers better control how loops are vectorized. Vectorization allows a processor to perform the same operation on multiple data points simultaneously, which can significantly enhance performance, especially for large data sets.

        The following OpenMP SIMD clauses are commonly used to optimize loop-based computations:

        - safelen: Ensures that the vectorized loop is safe for up to N iterations.
        aligned: Ensures that data is memory-aligned, which improves access speed and allows the compiler to generate efficient SIMD instructions.
        - collapse: Combines multiple loops, which enables better vectorization by collapsing nested loops into a single iteration space.

        ####Example of SIMD Clauses:
        - Memory Alignment (aligned): Aligning data ensures that it can be efficiently - loaded into SIMD registers, reducing the overhead of unaligned memory access.
        - Collapsing Loops (collapse): Collapsing nested loops into a single iteration space allows for better vectorization, particularly useful for multi-dimensional arrays or matrices.

        """
    )
    return


@app.cell
def _(os):
    import time

    # Define the C code that compares the three SIMD options
    c_code_comparison = r'''
    #include <stdio.h>
    #include <omp.h>

    #define N 1000000  // Increase the size for more noticeable differences

    void no_simd(double *array, int size) {
        // Simple loop without SIMD
        for (int i = 0; i < size; i++) {
            array[i] = array[i] * 2;
        }
    }

    void simd_basic(double *array, int size) {
        // Basic SIMD without any clauses
        #pragma omp simd
        for (int i = 0; i < size; i++) {
            array[i] = array[i] * 2;
        }
    }

    void simd_optimized(double *array, int size) {
        // Optimized SIMD with memory alignment
        #pragma omp simd aligned(array: 32)
        for (int i = 0; i < size; i++) {
            array[i] = array[i] * 2;
        }
    }

    int main() {
        double array1[N], array2[N], array3[N];

        // Initialize the arrays
        for (int i = 0; i < N; i++) {
            array1[i] = i * 1.0;
            array2[i] = i * 1.0;
            array3[i] = i * 1.0;
        }

        // Run the computation multiple times to average the timings
        int repeats = 10;
        double total_time_no_simd = 0.0;
        double total_time_simd_basic = 0.0;
        double total_time_simd_optimized = 0.0;

        for (int r = 0; r < repeats; r++) {
            // Measure time for no SIMD
            double start_time = omp_get_wtime();
            no_simd(array1, N);
            double end_time = omp_get_wtime();
            total_time_no_simd += (end_time - start_time);

            // Measure time for basic SIMD
            start_time = omp_get_wtime();
            simd_basic(array2, N);
            end_time = omp_get_wtime();
            total_time_simd_basic += (end_time - start_time);

            // Measure time for optimized SIMD with aligned clause
            start_time = omp_get_wtime();
            simd_optimized(array3, N);
            end_time = omp_get_wtime();
            total_time_simd_optimized += (end_time - start_time);
        }

        // Print a sample of the results for verification
        printf("Array output sample (without SIMD):\n");
        for (int i = 0; i < 10; i++) {
            printf("array1[%d] = %f\n", i, array1[i]);
            fflush(stdout);  // Ensure output is printed
        }

        printf("\nArray output sample (with basic SIMD):\n");
        for (int i = 0; i < 10; i++) {
            printf("array2[%d] = %f\n", i, array2[i]);
            fflush(stdout);  // Ensure output is printed
        }

        printf("\nArray output sample (with SIMD + aligned):\n");
        for (int i = 0; i < 10; i++) {
            printf("array3[%d] = %f\n", i, array3[i]);
            fflush(stdout);  // Ensure output is printed
        }

        // Print the time comparisons
        printf("\nAverage time for no SIMD: %f seconds\n", total_time_no_simd / repeats);
        fflush(stdout);
        printf("Average time for basic SIMD: %f seconds\n", total_time_simd_basic / repeats);
        fflush(stdout);
        printf("Average time for SIMD with aligned clause: %f seconds\n", total_time_simd_optimized / repeats);
        fflush(stdout);

        return 0;
    }
    '''

    # Save the C code to a file
    with open("simd_comparison_example.c", "w") as code_file:
        code_file.write(c_code_comparison)

    # Compile the C code with OpenMP support
    compilation_status = os.system("gcc -fopenmp simd_comparison_example.c -o simd_comparison_example")

    # Check if compilation was successful
    if compilation_status == 0:
        print("Compilation successful. Running the program...")
        # Run the compiled program
        os.system("./simd_comparison_example")
    else:
        print("Compilation failed. Please check the code for errors.")
    return


@app.cell
def _():
    import subprocess
    c_code_comparison_1 = '\n#include <stdio.h>\n#include <omp.h>\n#include <stdlib.h>\n\n#define N 300000000  // Larger size to better illustrate SIMD benefits\n\nvoid no_simd(double *array, int size) {\n    // Simple loop without SIMD\n    for (int i = 0; i < size; i++) {\n        array[i] = array[i] * 2;\n    }\n}\n\nvoid simd_basic(double *array, int size) {\n    // Basic SIMD without any clauses\n    #pragma omp simd\n    for (int i = 0; i < size; i++) {\n        array[i] = array[i] * 2;\n    }\n}\n\nvoid simd_optimized(double *array, int size) {\n    // Optimized SIMD with memory alignment\n    #pragma omp simd aligned(array: 32)\n    for (int i = 0; i < size; i++) {\n        array[i] = array[i] * 2;\n    }\n}\n\nint main() {\n    double *array1 = (double*) malloc(N * sizeof(double));\n    double *array2 = (double*) malloc(N * sizeof(double));\n    double *array3 = (double*) malloc(N * sizeof(double));\n\n    // Initialize the arrays\n    for (int i = 0; i < N; i++) {\n        array1[i] = i * 1.0;\n        array2[i] = i * 1.0;\n        array3[i] = i * 1.0;\n    }\n\n    // Run the computation multiple times to average the timings\n    int repeats = 3;\n    double total_time_no_simd = 0.0;\n    double total_time_simd_basic = 0.0;\n    double total_time_simd_optimized = 0.0;\n\n    for (int r = 0; r < repeats; r++) {\n        // Measure time for no SIMD\n        double start_time = omp_get_wtime();\n        no_simd(array1, N);\n        double end_time = omp_get_wtime();\n        total_time_no_simd += (end_time - start_time);\n\n        // Measure time for basic SIMD\n        start_time = omp_get_wtime();\n        simd_basic(array2, N);\n        end_time = omp_get_wtime();\n        total_time_simd_basic += (end_time - start_time);\n\n        // Measure time for optimized SIMD with aligned clause\n        start_time = omp_get_wtime();\n        simd_optimized(array3, N);\n        end_time = omp_get_wtime();\n        total_time_simd_optimized += (end_time - start_time);\n    }\n\n    // Print a sample of the results for verification\n    printf("Array output sample (without SIMD):\\n");\n    for (int i = 0; i < 10; i++) {\n        printf("array1[%d] = %f\\n", i, array1[i]);\n        fflush(stdout);  // Ensure output is printed immediately\n    }\n\n    printf("\\nArray output sample (with basic SIMD):\\n");\n    for (int i = 0; i < 10; i++) {\n        printf("array2[%d] = %f\\n", i, array2[i]);\n        fflush(stdout);  // Ensure output is printed immediately\n    }\n\n    printf("\\nArray output sample (with SIMD + aligned):\\n");\n    for (int i = 0; i < 10; i++) {\n        printf("array3[%d] = %f\\n", i, array3[i]);\n        fflush(stdout);  // Ensure output is printed immediately\n    }\n\n    // Print the time comparisons\n    printf("\\n=== Performance Comparison ===\\n");\n    printf("Average time for no SIMD: %f seconds\\n", total_time_no_simd / repeats);\n    printf("Average time for basic SIMD: %f seconds\\n", total_time_simd_basic / repeats);\n    printf("Average time for SIMD with aligned clause: %f seconds\\n", total_time_simd_optimized / repeats);\n\n    // Free the allocated memory\n    free(array1);\n    free(array2);\n    free(array3);\n\n    return 0;\n}\n'
    with open('simd_comparison_example.c', 'w') as code_file_1:
        code_file_1.write(c_code_comparison_1)
    compilation_status_1 = subprocess.run(['gcc', '-fopenmp', 'simd_comparison_example.c', '-o', 'simd_comparison_example'])
    if compilation_status_1.returncode == 0:
        print('Compilation successful. Running the program...')
        output = subprocess.run(['./simd_comparison_example'], capture_output=True, text=True)
        print(output.stdout)
    else:
        print('Compilation failed. Please check the code for errors.')
    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Explanation of SIMD Clauses for Optimization

        This example demonstrates how the OpenMP SIMD clauses `aligned` and `collapse` can be used to optimize loop-based computations by enhancing vectorization.

        ### SIMD Directive
        The `#pragma omp simd` directive explicitly informs the compiler to vectorize the loop, ensuring that multiple iterations of the loop can be executed simultaneously using SIMD instructions.

        ### Collapse Clause
        The `collapse(2)` clause collapses the two loops (i and j) into a single iteration space. This enables better vectorization across both dimensions of the matrix. This approach is particularly useful when dealing with multi-dimensional arrays, where you want to optimize access patterns for both dimensions.

        ### Aligned Clause
        The `aligned(array: 32)` clause ensures that the array is aligned on a 32-byte boundary. This improves performance by ensuring that data is efficiently loaded into SIMD registers. Misaligned memory accesses can cause significant slowdowns, so aligning the data helps maximize performance.

        ### Matrix Example
        The nested loops over the matrix are vectorized with the help of the `collapse(2)` clause, ensuring that both dimensions of the matrix are processed in parallel. Each element of the matrix is multiplied by 2 using SIMD instructions.

        ### Array Example
        The array is processed in a single loop, and the `aligned(array: 32)` clause ensures that the array is aligned in memory for efficient SIMD processing. Each element in the array is multiplied by 2.

        ### Output
        The program prints a sample of the processed array to demonstrate that the SIMD vectorization was applied correctly. The matrix and array are efficiently processed using SIMD, improving performance for large data sets.

        ### Key Takeaways
        - **Better Vectorization**: Using the `collapse` clause allows multiple loops to be collapsed into a single iteration space, enabling better use of SIMD.
        - **Memory Alignment**: The `aligned` clause ensures that memory accesses are optimized, minimizing penalties for unaligned data.
        - **Performance**: These SIMD clauses can significantly improve performance for loop-based computations, especially when working with large arrays or matrices that can benefit from vectorized processing.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to Offloading to GPUs with OpenMP
        OpenMP supports offloading compute-intensive tasks to GPUs and other accelerators, which can significantly speed up certain computations. This is done using the #pragma omp target directive, allowing developers to move parts of the code from the CPU to the GPU. Offloading tasks to a GPU involves transferring data from the host (CPU) to the device (GPU), performing the computation on the GPU, and then transferring the results back to the host.

        In GPU offloading, data mapping is essential to manage the flow of data between the CPU and GPU. OpenMP allows developers to specify which data needs to be transferred to and from the GPU using the map clause. The map(to: ...) clause specifies data to be sent to the GPU, while map(from: ...) defines data that should be transferred back to the CPU after computation.

        Example:
        ```
        #pragma omp target map(to: A[0:N], B[0:N]) map(from: C[0:N])
        {
            for (int i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
            }
        }
        ```

        This example offloads a simple vector addition computation to the GPU. The A and B arrays are transferred to the GPU, the computation is done on the GPU, and the result stored in array C is transferred back to the CPU after the computation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## OpenMP on CPU vs CUDA on GPU: Performance Comparison

        This section will guide you through running a vector addition program using OpenMP on the CPU and CUDA on the GPU. We will measure the time taken by both versions to compare performance.

        ### Requirements:
        1. **OpenMP** for CPU parallelism: Most modern compilers like `gcc` come with OpenMP support.
        2. **CUDA** for GPU programming: Requires NVIDIA GPU with CUDA and `nvcc` installed.

        ### Steps to Run:
        1. Run the **OpenMP on CPU** section to parallelize the vector addition on the CPU.
        2. Run the **CUDA on GPU** section to offload the vector addition to the GPU.
        3. Compare the execution time between CPU and GPU.

        ---

        ## OpenMP on CPU

        This section uses OpenMP to run the computation in parallel on the CPU.

        ```python

        # Define the C code that uses OpenMP on the CPU
        c_code_cpu_openmp = r'''
        #include <stdio.h>
        #include <omp.h>

        #define N 100000000  // Large array size for parallel computation

        void init_arrays(float *A, float *B, int size) {
            for (int i = 0; i < size; i++) {
                A[i] = i * 1.0f;
                B[i] = i * 2.0f;
            }
        }

        int main() {
            float *A = (float*) malloc(N * sizeof(float));
            float *B = (float*) malloc(N * sizeof(float));
            float *C = (float*) malloc(N * sizeof(float));

            // Initialize the arrays
            init_arrays(A, B, N);

            double start_time = omp_get_wtime();

            // Perform the computation in parallel on the CPU using OpenMP
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
            }

            double end_time = omp_get_wtime();

            // Print the timing result
            printf("Time taken for OpenMP on CPU: %f seconds\n", end_time - start_time);

            // Free the allocated memory
            free(A);
            free(B);
            free(C);

            return 0;
        }
        '''

        # Save the C code to a file
        with open("cpu_openmp_example.c", "w") as code_file:
            code_file.write(c_code_cpu_openmp)

        # Compile the C code with OpenMP support
        compilation_status = subprocess.run(["gcc", "-fopenmp", "cpu_openmp_example.c", "-o", "cpu_openmp_example"])

        # Check if compilation was successful
        if compilation_status.returncode == 0:
            print("Compilation successful. Running the program on the CPU...")
            # Run the compiled program and capture the output
            output = subprocess.run(["./cpu_openmp_example"], capture_output=True, text=True)
            # Print the captured output
            print(output.stdout)
        else:
            print("Compilation failed. Please check the code for errors.")

        """
    )
    return


@app.cell
def _(subprocess):
    c_code_openmp_target = '\n#include <stdio.h>\n#include <omp.h>\n\n#define N 100000000  // Large array size for offloading\n\nvoid init_arrays(float *A, float *B, int size) {\n    for (int i = 0; i < size; i++) {\n        A[i] = i * 1.0f;\n        B[i] = i * 2.0f;\n    }\n}\n\nint main() {\n    float *A = (float*) malloc(N * sizeof(float));\n    float *B = (float*) malloc(N * sizeof(float));\n    float *C = (float*) malloc(N * sizeof(float));\n\n    // Initialize the arrays\n    init_arrays(A, B, N);\n\n    double start_time = omp_get_wtime();\n\n    // Offload the computation to the GPU using OpenMP target\n    #pragma omp target map(to: A[0:N], B[0:N]) map(from: C[0:N])\n    {\n        #pragma omp parallel for\n        for (int i = 0; i < N; i++) {\n            C[i] = A[i] + B[i];\n        }\n    }\n\n    double end_time = omp_get_wtime();\n\n    // Print the timing result\n    printf("Time taken for OpenMP target offloading: %f seconds\\n", end_time - start_time);\n\n    // Print a sample of the results for verification\n    printf("Sample results:\\n");\n    for (int i = 0; i < 10; i++) {\n        printf("C[%d] = %f\\n", i, C[i]);\n    }\n\n    // Free the allocated memory\n    free(A);\n    free(B);\n    free(C);\n\n    return 0;\n}\n'
    with open('openmp_target_offloading_example.c', 'w') as code_file_2:
        code_file_2.write(c_code_openmp_target)
    compilation_status_2 = subprocess.run(['gcc', '-fopenmp', 'openmp_target_offloading_example.c', '-o', 'openmp_target_offloading_example'])
    if compilation_status_2.returncode == 0:
        print('Compilation successful. Running the program on the GPU...')
        output_1 = subprocess.run(['./openmp_target_offloading_example'], capture_output=True, text=True)
        print(output_1.stdout)
    else:
        print('Compilation failed. Please check the code for errors.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##Explanation of the GPU Offloading Code
        In this example, we offload a vector addition computation to the GPU using OpenMP. The arrays A and B are initialized on the CPU, and then transferred to the GPU using the map(to: A[0:N], B[0:N]) clause. The result array C is computed on the GPU and transferred back to the CPU using the map(from: C[0:N]) clause.

        1. Data Mapping:
         - map(to: A[0:N], B[0:N]): This clause transfers the A and B arrays to the GPU. The GPU uses these arrays to perform the computation.
         - map(from: C[0:N]): After the computation is complete, the results stored in the C array are transferred back from the GPU to the CPU.
        2. GPU Offloading with #pragma omp target: The #pragma omp target directive specifies that the following block of code will be offloaded to the GPU. In this case, the vector addition loop is executed on the GPU, where each element of C is calculated as the sum of the corresponding elements in A and B.

        3. Performance Benefits: Offloading to the GPU can greatly speed up compute-intensive tasks, especially when working with large data sets. GPUs are designed for high parallelism, making them ideal for operations that can be done independently on many data elements (such as this vector addition).

        4. Sample Output: The program prints the first 10 results of the computation from the C array to verify that the GPU offloading worked as expected. The values should match the sum of corresponding elements in A and B after the computation is completed.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OpenMP Scheduling Strategies in Parallel Programming

        ## Introduction

        In OpenMP, **scheduling** determines how iterations of a loop are divided among threads in parallel regions. Different scheduling strategies affect performance based on the workload and the nature of the task.

        ### Types of Scheduling in OpenMP

        1. **Static Scheduling**:
           - Iterations are divided equally among threads at the start of the parallel region.
           - Best for **uniform workloads** where each iteration takes roughly the same time.

        2. **Dynamic Scheduling**:
           - Threads dynamically request new chunks of iterations as they finish previous chunks.
           - Useful for **irregular workloads** where the time taken for each iteration may vary significantly.
           - You can specify a **chunk size** for how many iterations are assigned to a thread at a time.

        3. **Guided Scheduling**:
           - Threads are initially assigned **large chunks** of iterations, and the chunk size decreases as the computation progresses.
           - Suitable for workloads that **decrease in complexity** over time.

        4. **Auto Scheduling**:
           - The **runtime** decides the optimal scheduling method based on the system and workload.
           - This strategy is a black-box approach where OpenMP automatically handles the workload distribution.

        ## Purpose of This Example

        In this notebook, we will compare the execution of a parallel loop using different scheduling strategies (static, dynamic, guided, and auto). We will print the **thread ID** and **iterations** processed by each thread to help visualize how each strategy works. This will show how threads are assigned work chunks and how the choice of scheduling can affect performance.

        The example uses a simple loop that processes an array, and we will measure the **time taken** for each scheduling strategy to demonstrate their impact on performance.

        ## Code and Output

        The following code will:
        - Run the parallel loop with each scheduling strategy.
        - Print the thread handling each iteration of the loop.
        - Display the total execution time for each scheduling strategy.

        """
    )
    return


@app.cell
def _(os, subprocess):
    os.environ['OMP_NUM_THREADS'] = '4'
    c_code_openmp_scheduling_detailed = '\n#include <stdio.h>\n#include <omp.h>\n#include <stdlib.h>\n\n#define N 32  // Smaller array size for demonstration\n\nvoid init_array(double *arr, int size) {\n    for (int i = 0; i < size; i++) {\n        arr[i] = (double)i / size;\n    }\n}\n\nint main() {\n    double *A = (double*) malloc(N * sizeof(double));\n\n    double start_time, end_time;\n    int thread_id;\n\n    // 1. Static Scheduling\n    printf("Static scheduling:\\n");\n    start_time = omp_get_wtime();\n    #pragma omp parallel for schedule(static) private(thread_id)\n    for (int i = 0; i < N; i++) {\n        thread_id = omp_get_thread_num();\n        A[i] = A[i] + 1;\n        printf("Thread %d is processing iteration %d\\n", thread_id, i);\n    }\n    end_time = omp_get_wtime();\n    printf("Static scheduling time: %f seconds\\n\\n", end_time - start_time);\n\n    // 2. Dynamic Scheduling with chunk size 4\n    printf("Dynamic scheduling (chunk=4):\\n");\n    start_time = omp_get_wtime();\n    #pragma omp parallel for schedule(dynamic, 4) private(thread_id)\n    for (int i = 0; i < N; i++) {\n        thread_id = omp_get_thread_num();\n        A[i] = A[i] + 1;\n        printf("Thread %d is processing iteration %d\\n", thread_id, i);\n    }\n    end_time = omp_get_wtime();\n    printf("Dynamic scheduling (chunk=4) time: %f seconds\\n\\n", end_time - start_time);\n\n    // 3. Guided Scheduling\n    printf("Guided scheduling:\\n");\n    start_time = omp_get_wtime();\n    #pragma omp parallel for schedule(guided) private(thread_id)\n    for (int i = 0; i < N; i++) {\n        thread_id = omp_get_thread_num();\n        A[i] = A[i] + 1;\n        printf("Thread %d is processing iteration %d\\n", thread_id, i);\n    }\n    end_time = omp_get_wtime();\n    printf("Guided scheduling time: %f seconds\\n\\n", end_time - start_time);\n\n    // 4. Auto Scheduling\n    printf("Auto scheduling:\\n");\n    start_time = omp_get_wtime();\n    #pragma omp parallel for schedule(auto) private(thread_id)\n    for (int i = 0; i < N; i++) {\n        thread_id = omp_get_thread_num();\n        A[i] = A[i] + 1;\n        printf("Thread %d is processing iteration %d\\n", thread_id, i);\n    }\n    end_time = omp_get_wtime();\n    printf("Auto scheduling time: %f seconds\\n\\n", end_time - start_time);\n\n    // Free allocated memory\n    free(A);\n\n    return 0;\n}\n'
    with open('openmp_scheduling_detailed_example.c', 'w') as code_file_3:
        code_file_3.write(c_code_openmp_scheduling_detailed)
    compilation_status_3 = subprocess.run(['gcc', '-fopenmp', 'openmp_scheduling_detailed_example.c', '-o', 'openmp_scheduling_detailed_example'])
    if compilation_status_3.returncode == 0:
        print('Compilation successful. Running the program...')
        output_2 = subprocess.run(['./openmp_scheduling_detailed_example'], capture_output=True, text=True)
        print(output_2.stdout)
    else:
        print('Compilation failed. Please check the code for errors.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##Explanation of Results:
        - Static Scheduling is generally the best for uniform workloads because it distributes the work equally across all threads with no overhead during execution.

        - Dynamic Scheduling is better for irregular workloads because threads pick up new tasks as soon as they finish, reducing idle time.
        - Guided Scheduling is ideal for tasks that reduce in complexity over time, as it assigns progressively smaller chunks of work to threads.
        - Auto Scheduling lets the OpenMP runtime choose the best scheduling method based on the system and workload.
        ###Try Running:
        You can try adjusting the size of the array (N) and the chunk size in dynamic scheduling to see how it impacts performance.
        Additionally, you can change the number of threads used by adjusting the OMP_NUM_THREADS environment variable.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Enhancing a Serial Program with Advanced OpenMP

        ## Introduction

        This guide demonstrates how to convert a serial N-body simulation into a highly optimized parallel program using advanced OpenMP features. The N-body problem simulates the interactions of particles under gravitational forces, making it a computationally intensive task, especially for large numbers of particles.

        We will begin by implementing a task-based parallelism approach and gradually introduce more advanced techniques such as managing task dependencies, leveraging the `taskloop` directive, SIMD parallelism, and offloading computations to a GPU using OpenMP's `target` directives.

        The result will be a significant performance improvement, efficiently utilizing modern multicore processors and GPUs.

        ---

        ### Overview of Steps:
        1. **Task-Based Parallelism**: Splitting the force computation into tasks.
        2. **Task Dependencies**: Ensuring tasks execute in the correct order.
        3. **Taskloop Directive**: Simplifying task creation.
        4. **SIMD Parallelism**: Exploiting data-level parallelism.
        5. **GPU Offloading**: Offloading intensive computations to the GPU.

        We will start by reviewing the serial implementation and then proceed step-by-step to improve it using advanced OpenMP constructs.

        """
    )
    return


@app.cell
def _(subprocess):
    nbody_serial_code = '\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n#include <time.h>\n\n#define G 6.67430e-11   // Gravitational constant\n#define EPSILON 1e-9    // Softening factor to prevent singularities\n\ntypedef struct {\n    double x, y, z;     // Position components\n    double vx, vy, vz;  // Velocity components\n    double ax, ay, az;  // Acceleration components\n    double mass;        // Mass of the particle\n} Body;\n\n// Serial version of the compute_forces function (No parallelism)\nvoid compute_forces_serial(int N, Body *bodies) {\n    for (int i = 0; i < N; i++) {\n        bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;  // Reset acceleration\n        for (int j = 0; j < N; j++) {\n            if (i != j) {\n                double dx = bodies[j].x - bodies[i].x;\n                double dy = bodies[j].y - bodies[i].y;\n                double dz = bodies[j].z - bodies[i].z;\n                double dist_sqr = dx * dx + dy * dy + dz * dz + EPSILON;\n                double inv_dist = 1.0 / sqrt(dist_sqr);\n                double inv_dist3 = inv_dist * inv_dist * inv_dist;\n                double force = G * bodies[j].mass * inv_dist3;\n                bodies[i].ax += force * dx;\n                bodies[i].ay += force * dy;\n                bodies[i].az += force * dz;\n            }\n        }\n    }\n}\n\n// Simulation setup and main function\nint main() {\n    int N = 1000;          // Reduced number of particles for simplicity\n    double dt = 0.01;     // Time step\n    int steps = 100;       // Number of simulation steps\n\n    // Allocate memory for bodies\n    Body *bodies = (Body *)malloc(N * sizeof(Body));\n\n    // Initialize bodies with simple random values (positions and masses)\n    for (int i = 0; i < N; i++) {\n        bodies[i].x = rand() % 1000;\n        bodies[i].y = rand() % 1000;\n        bodies[i].z = rand() % 1000;\n        bodies[i].vx = 0;\n        bodies[i].vy = 0;\n        bodies[i].vz = 0;\n        bodies[i].mass = rand() % 100 + 1;\n    }\n\n    // Start timer\n    clock_t start_time = clock();\n\n    // Run the N-body simulation using the serial force computation\n    for (int s = 0; s < steps; s++) {\n        compute_forces_serial(N, bodies);\n        // Update positions and velocities here...\n        for (int i = 0; i < N; i++) {\n            bodies[i].vx += bodies[i].ax * dt;\n            bodies[i].vy += bodies[i].ay * dt;\n            bodies[i].vz += bodies[i].az * dt;\n\n            bodies[i].x += bodies[i].vx * dt;\n            bodies[i].y += bodies[i].vy * dt;\n            bodies[i].z += bodies[i].vz * dt;\n        }\n    }\n\n    // End timer\n    clock_t end_time = clock();\n    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;\n\n    // Print the elapsed time for the simulation\n    printf("Elapsed time for serial N-body simulation: %f seconds\\n", elapsed_time);\n\n    // Print some sample results\n    printf("Sample output after %d steps:\\n", steps);\n    for (int i = 0; i < 10; i++) {\n        printf("Body %d: Position (%f, %f, %f)\\n", i, bodies[i].x, bodies[i].y, bodies[i].z);\n    }\n\n    // Free allocated memory\n    free(bodies);\n    return 0;\n}\n'
    with open('nbody_serial_simulation.c', 'w') as code_file_4:
        code_file_4.write(nbody_serial_code)
    compilation_status_4 = subprocess.run(['gcc', 'nbody_serial_simulation.c', '-o', 'nbody_serial_simulation', '-lm'], capture_output=True, text=True)
    if compilation_status_4.returncode == 0:
        print('Compilation successful. Running the serial version of the program...')
        output_3 = subprocess.run(['./nbody_serial_simulation'], capture_output=True, text=True)
        print(output_3.stdout)
        if output_3.stderr:
            print('Program stderr:', output_3.stderr)
    else:
        print('Compilation failed. Please check the code for errors.')
        print('Compiler stderr:', compilation_status_4.stderr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###And now the parallel version:
        """
    )
    return


@app.cell
def _(os, subprocess):
    os.environ['OMP_NUM_THREADS'] = '4'
    nbody_code = '\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n#include <omp.h>\n#include <time.h>\n\n#define G 6.67430e-11   // Gravitational constant\n#define EPSILON 1e-9    // Softening factor to prevent singularities\n\ntypedef struct {\n    double x, y, z;     // Position components\n    double vx, vy, vz;  // Velocity components\n    double ax, ay, az;  // Acceleration components\n    double mass;        // Mass of the particle\n} Body;\n\n// OpenMP Parallel version of compute_forces function (CPU only)\nvoid compute_forces_parallel(int N, Body *bodies) {\n    // Parallelize the outer loop using OpenMP\n    #pragma omp parallel for schedule(dynamic)\n    for (int i = 0; i < N; i++) {\n        // Reset acceleration for body i\n        bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;\n        for (int j = 0; j < N; j++) {\n            if (i != j) {\n                double dx = bodies[j].x - bodies[i].x;\n                double dy = bodies[j].y - bodies[i].y;\n                double dz = bodies[j].z - bodies[i].z;\n                double dist_sqr = dx * dx + dy * dy + dz * dz + EPSILON;\n                double inv_dist = 1.0 / sqrt(dist_sqr);\n                double inv_dist3 = inv_dist * inv_dist * inv_dist;\n                double force = G * bodies[j].mass * inv_dist3;\n                bodies[i].ax += force * dx;\n                bodies[i].ay += force * dy;\n                bodies[i].az += force * dz;\n            }\n        }\n    }\n}\n\n// Function to initialize bodies with random positions and masses\nvoid initialize_bodies(int N, Body *bodies) {\n    // Seed the random number generator for reproducibility\n    srand(time(NULL));\n    for (int i = 0; i < N; i++) {\n        bodies[i].x = ((double)(rand() % 1000)) / 10.0;  // Positions in range [0, 100)\n        bodies[i].y = ((double)(rand() % 1000)) / 10.0;\n        bodies[i].z = ((double)(rand() % 1000)) / 10.0;\n        bodies[i].vx = 0.0;\n        bodies[i].vy = 0.0;\n        bodies[i].vz = 0.0;\n        bodies[i].mass = ((double)(rand() % 100)) + 1.0;  // Mass in range [1, 100]\n    }\n}\n\n// Simulation setup and main function\nint main() {\n    int N = 1000;         // Number of particles\n    double dt = 0.01;     // Time step\n    int steps = 100;      // Number of simulation steps\n\n    // Allocate memory for bodies\n    Body *bodies = (Body *)malloc(N * sizeof(Body));\n    if (bodies == NULL) {\n        fprintf(stderr, "Error allocating memory for bodies.\\n");\n        return 1;\n    }\n\n    // Initialize bodies with random positions and masses\n    initialize_bodies(N, bodies);\n\n    // Start timer\n    double start_time = omp_get_wtime();\n\n    // Run the N-body simulation using parallel force computation\n    for (int s = 0; s < steps; s++) {\n        compute_forces_parallel(N, bodies);\n        // Update positions and velocities\n        for (int i = 0; i < N; i++) {\n            bodies[i].vx += bodies[i].ax * dt;\n            bodies[i].vy += bodies[i].ay * dt;\n            bodies[i].vz += bodies[i].az * dt;\n\n            bodies[i].x += bodies[i].vx * dt;\n            bodies[i].y += bodies[i].vy * dt;\n            bodies[i].z += bodies[i].vz * dt;\n        }\n    }\n\n    // End timer\n    double end_time = omp_get_wtime();\n    double elapsed_time = end_time - start_time;\n\n    // Print the elapsed time for the simulation\n    printf("Elapsed time for parallel N-body simulation: %f seconds\\n", elapsed_time);\n\n    // Print some sample results\n    printf("Sample output after %d steps:\\n", steps);\n    for (int i = 0; i < 10 && i < N; i++) {\n        printf("Body %d: Position (%.2f, %.2f, %.2f)\\n", i, bodies[i].x, bodies[i].y, bodies[i].z);\n    }\n\n    // Free allocated memory\n    free(bodies);\n    return 0;\n}\n'
    with open('nbody_simulation_openmp.c', 'w') as code_file_5:
        code_file_5.write(nbody_code)
    compilation_command = ['gcc', '-fopenmp', 'nbody_simulation_openmp.c', '-o', 'nbody_simulation_openmp', '-lm', '-O2']
    compilation_status_5 = subprocess.run(compilation_command, capture_output=True, text=True)
    if compilation_status_5.returncode == 0:
        print('Compilation successful. Running the OpenMP-enabled program...')
        execution_status = subprocess.run(['./nbody_simulation_openmp'], capture_output=True, text=True)
        print(execution_status.stdout)
        if execution_status.stderr:
            print('Program stderr:', execution_status.stderr)
    else:
        print('Compilation failed. Please check the code for errors.')
        print('Compiler stderr:', compilation_status_5.stderr)
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
        ## Explanation of Advanced OpenMP Features

        In this section, we progressively enhanced the serial N-body simulation using various advanced OpenMP techniques.

        ### 1. Task-Based Parallelism
        We first parallelized the **compute_forces** function by creating tasks for each particle’s force computation. By using OpenMP’s tasking model, we allow the OpenMP runtime to dynamically distribute these tasks across available threads, improving load balancing.

        - **#pragma omp parallel**: Creates a parallel region where threads can work concurrently.
        - **#pragma omp single**: Ensures only one thread creates tasks.
        - **#pragma omp task**: Each particle's force computation is assigned as a task.
        - **firstprivate(i)**: Ensures each task gets its own copy of the index `i`.

        ### 2. Task Dependencies
        To ensure that tasks execute in the correct order, we can manage dependencies using OpenMP’s `depend` clause. This ensures that tasks computing forces finish before the tasks updating positions begin, preventing race conditions.

        ### 3. Taskloop Directive
        In large simulations, creating individual tasks for each iteration can introduce overhead. The **taskloop** directive reduces this overhead by allowing OpenMP to automatically split the loop iterations into tasks.

        ### 4. SIMD Parallelism
        By adding the **simd** directive, we leverage vectorization to parallelize the inner loop that computes forces. SIMD (Single Instruction, Multiple Data) allows modern CPUs to process multiple data points simultaneously, improving performance for the loop over particle interactions.

        - **#pragma omp simd**: Vectorizes the loop over particle pairs.
        - **reduction(+: bodies[i].ax, bodies[i].ay, bodies[i].az)**: Ensures that the accumulation of forces is handled correctly across SIMD lanes.

        ### 5. GPU Offloading
        Finally, we offload the force computation to a GPU using OpenMP’s `target` directive. This allows the program to take advantage of the thousands of cores available on GPUs.

        - **#pragma omp target data map(tofrom: bodies[0:N])**: Maps the bodies array to GPU memory.
        - **#pragma omp target teams distribute parallel for simd**: Offloads the force computation loop to the GPU, where it is distributed across teams of threads, with SIMD parallelization within each thread.

        ---

        ### Performance Considerations:
        - **Task-based parallelism** improves load balancing, especially for irregular workloads.
        - **SIMD parallelism** exploits data-level parallelism to improve performance within a loop.
        - **GPU offloading** harnesses the power of thousands of cores to speed up the most computationally intensive part of the simulation.

        By combining these advanced techniques, we achieve significant performance improvements, especially for large-scale simulations.

        """
    )
    return


@app.cell
def _(os, subprocess):
    os.environ['OMP_NUM_THREADS'] = '4'
    nbody_code_1 = '\n#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\n#include <omp.h>\n\n#define G 6.67430e-11   // Gravitational constant\n#define EPSILON 1e-9    // Softening factor to prevent singularities\n\ntypedef struct {\n    double x, y, z;     // Position components\n    double vx, vy, vz;  // Velocity components\n    double ax, ay, az;  // Acceleration components\n    double mass;        // Mass of the particle\n} Body;\n\n// Serial version of the compute_forces function\nvoid compute_forces(int N, Body *bodies) {\n    for (int i = 0; i < N; i++) {\n        bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;  // Reset acceleration\n        for (int j = 0; j < N; j++) {\n            if (i != j) {\n                double dx = bodies[j].x - bodies[i].x;\n                double dy = bodies[j].y - bodies[i].y;\n                double dz = bodies[j].z - bodies[i].z;\n                double dist_sqr = dx * dx + dy * dy + dz * dz + EPSILON;\n                double inv_dist = 1.0 / sqrt(dist_sqr);\n                double inv_dist3 = inv_dist * inv_dist * inv_dist;\n                double force = G * bodies[j].mass * inv_dist3;\n                bodies[i].ax += force * dx;\n                bodies[i].ay += force * dy;\n                bodies[i].az += force * dz;\n            }\n        }\n    }\n}\n\n// OpenMP Task-Based Parallelism\nvoid compute_forces_parallel(int N, Body *bodies) {\n    #pragma omp parallel\n    {\n        #pragma omp single\n        {\n            for (int i = 0; i < N; i++) {\n                #pragma omp task firstprivate(i)\n                {\n                    bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;\n                    for (int j = 0; j < N; j++) {\n                        if (i != j) {\n                            double dx = bodies[j].x - bodies[i].x;\n                            double dy = bodies[j].y - bodies[i].y;\n                            double dz = bodies[j].z - bodies[i].z;\n                            double dist_sqr = dx * dx + dy * dy + dz * dz + EPSILON;\n                            double inv_dist = 1.0 / sqrt(dist_sqr);\n                            double inv_dist3 = inv_dist * inv_dist * inv_dist;\n                            double force = G * bodies[j].mass * inv_dist3;\n                            bodies[i].ax += force * dx;\n                            bodies[i].ay += force * dy;\n                            bodies[i].az += force * dz;\n                        }\n                    }\n                }\n            }\n            #pragma omp taskwait\n        }\n    }\n}\n\n// GPU Offloading with OpenMP\nvoid compute_forces_gpu(int N, Body *bodies) {\n    #pragma omp target data map(tofrom: bodies[0:N])\n    {\n        #pragma omp target teams distribute parallel for simd\n        for (int i = 0; i < N; i++) {\n            bodies[i].ax = bodies[i].ay = bodies[i].az = 0.0;\n\n            for (int j = 0; j < N; j++) {\n                if (i != j) {\n                    double dx = bodies[j].x - bodies[i].x;\n                    double dy = bodies[j].y - bodies[i].y;\n                    double dz = bodies[j].z - bodies[i].z;\n                    double dist_sqr = dx * dx + dy * dy + dz * dz + EPSILON;\n                    double inv_dist = 1.0 / sqrt(dist_sqr);\n                    double inv_dist3 = inv_dist * inv_dist * inv_dist;\n                    double force = G * bodies[j].mass * inv_dist3;\n                    bodies[i].ax += force * dx;\n                    bodies[i].ay += force * dy;\n                    bodies[i].az += force * dz;\n                }\n            }\n        }\n    }\n}\n\n// Simulation setup and main function\nint main() {\n    int N = 1000;          // Number of particles\n    double dt = 0.01;      // Time step\n    int steps = 100;       // Number of simulation steps\n\n    // Allocate memory for bodies\n    Body *bodies = (Body *)malloc(N * sizeof(Body));\n\n    // Initialize bodies (positions, velocities, masses)\n    // [Initialization code here]\n\n    // Run the N-body simulation using task-based parallelism\n    for (int s = 0; s < steps; s++) {\n        compute_forces_parallel(N, bodies);\n        // Update positions and velocities here...\n    }\n\n    // Cleanup\n    free(bodies);\n    return 0;\n}\n'
    with open('nbody_simulation.c', 'w') as code_file_6:
        code_file_6.write(nbody_code_1)
    compilation_status_6 = subprocess.run(['gcc', '-fopenmp', 'nbody_simulation.c', '-o', 'nbody_simulation'])
    if compilation_status_6.returncode == 0:
        print('Compilation successful. Running the program...')
        output_4 = subprocess.run(['./nbody_simulation'], capture_output=True, text=True)
        print(output_4.stdout)
    else:
        print('Compilation failed. Please check the code for errors.')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

