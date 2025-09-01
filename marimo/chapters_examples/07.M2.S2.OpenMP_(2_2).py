import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise 7.1: Introduction to OpenMP

        OpenMP (Open Multi-Processing) is a powerful API designed for parallel programming in shared-memory environments. This exercise will introduce you to the basics of OpenMP, including how to use OpenMP directives to parallelize loops and how to manage thread parallelism.

        ## 7.1.1 Overview of OpenMP

        ### 7.1.1.1 Definition and Purpose of OpenMP
        OpenMP provides a simple and flexible interface for developing parallel applications by using compiler directives, runtime library routines, and environment variables. It allows developers to parallelize existing serial code incrementally, making it easier to transition from sequential to parallel programming.

        ### 7.1.1.2 Historical Context and Development
        OpenMP was first introduced in 1997 and has since evolved with support for task-based parallelism, accelerator directives, and memory management improvements, making it a relevant tool in modern HPC environments.

        ### 7.1.1.3 Applicability in Modern HPC Environments
        OpenMP is widely applicable in modern HPC due to its ability to leverage multicore architectures efficiently. It is used in scientific simulations, data analysis, and real-time processing.

        ## 7.1.2 Key Features of OpenMP

        ### 7.1.2.1 Simple and Flexible Parallel Programming Model
        OpenMP simplifies parallel programming by allowing developers to parallelize loops with minimal code changes. For example, using the `#pragma omp parallel for` directive to parallelize a loop.

        ### 7.1.2.2 Support for C, C++, and Fortran
        OpenMP supports multiple programming languages, including C, C++, and Fortran, which broadens its applicability across various scientific and engineering domains.

        ### 7.1.2.3 Portable Across Different Shared-Memory Architectures
        OpenMP is portable across various shared-memory architectures, ensuring that parallel code can run efficiently on different systems without modification.

        ## 7.1.3 Installation and Setup of OpenMP

        ### 7.1.3.1 Installing OpenMP on Various Platforms
        - **Linux:** Use GCC with the `-fopenmp` flag to compile OpenMP programs.
        - **Windows:** Use MinGW or Visual Studio to enable OpenMP support.
        - **MacOS:** Use Homebrew to install GCC for OpenMP support.

        ### 7.1.3.2 Compiler Support for OpenMP
        OpenMP is supported by GCC, Clang, and Intel Compilers, each providing robust support for parallel programming.

        ## 7.1.3.3 Thread Parallelism
        Thread parallelism in OpenMP divides a task into smaller sub-tasks that can be executed by multiple threads simultaneously. This approach leverages multicore processors for efficient parallel execution.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this program, we use `#pragma omp parallel` to instruct the compiler to parallelize the enclosed code block. The `omp_get_thread_num()` function returns the thread number executing the current block, which helps us verify that multiple threads are indeed running the code concurrently.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing a simple C code for OpenMP Hello World
    hello_world_code = \"\"\"
    #include <omp.h>
    #include <stdio.h>

    int main() {
        // Parallel region with OpenMP
        #pragma omp parallel
        {
            // Each thread prints \"Hello World\"
            printf(\"Hello World from thread %d\\n\", omp_get_thread_num());
        }
        return 0;
    }
    \"\"\"

    # Saving the code to a file
    with open(\"hello_world.c\", \"w\") as f:
        f.write(hello_world_code)

    # Compile the program with OpenMP support
    !gcc -fopenmp hello_world.c -o hello_world

    # Run the program
    !./hello_world
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Writing a simple C code for OpenMP Hello World with thread count input
    hello_world_code = \"\"\"
    #include <omp.h>
    #include <stdio.h>

    int main() {
        int num_threads;

        // Prompting the user to input the number of threads
        printf(\"Enter the number of threads: \");
        scanf(\"%d\", &num_threads);

        // Set the number of threads for OpenMP
        omp_set_num_threads(num_threads);

        // Parallel region with OpenMP
        #pragma omp parallel
        {
            // Each thread prints \"Hello World\"
            printf(\"Hello World from thread %d out of %d\\n\", omp_get_thread_num(), omp_get_num_threads());
        }
        return 0;
    }
    \"\"\"

    # Saving the code to a file
    with open(\"hello_world.c\", \"w\") as f:
        f.write(hello_world_code)

    # Compile the program with OpenMP support
    !gcc -fopenmp hello_world.c -o hello_world

    # Run the program (students will input the number of threads at runtime)
    !./hello_world
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
        In this example, we parallelize a loop that computes the sum of the first N numbers. We use the `#pragma omp parallel for` directive to split the loop iterations across different threads. The `reduction(+:sum)` clause ensures that the partial sums from each thread are safely combined into a final result.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Writing a simple OpenMP program to parallelize a loop
    loop_code = \"\"\"
    #include <omp.h>
    #include <stdio.h>

    int main() {
        int N = 1000;
        int sum = 0;

        // Parallelize this loop
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++) {
            sum += i;
        }

        printf(\"Sum of first %d numbers is %d\\n\", N, sum);
        return 0;
    }
    \"\"\"

    # Save the code to a file
    with open(\"parallel_loop.c\", \"w\") as f:
        f.write(loop_code)

    # Compile the program with OpenMP support
    !gcc -fopenmp parallel_loop.c -o parallel_loop

    # Run the program
    !./parallel_loop
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Serial vs Parallel Code Using OpenMP

        In this notebook, we will compare two versions of a simple program that adds elements of two arrays. The first version is a serial implementation, where the entire task is done sequentially by a single thread. The second version is a parallel implementation using OpenMP, where the array operations are divided among multiple threads.

        Understanding the difference between serial and parallel execution is crucial in HPC, as it demonstrates how parallelization can improve performance by utilizing multiple CPU cores simultaneously.

        ### **Serial Code Explanation**

        In the serial code, we create two arrays `a[]` and `b[]`, initialize them, and then add their corresponding elements to form a result array `result[]`. The entire process is done sequentially, and only one thread (the main thread) performs the operations.

        ### **Parallel Code Explanation**

        In the parallel code, we use OpenMP to divide the array addition task among multiple threads. Each thread performs the addition for a different portion of the arrays. This is done using the `#pragma omp parallel` directive, which forks multiple threads, and `#pragma omp for`, which divides the loop iterations among those threads.

        Parallelization helps reduce execution time when dealing with larger datasets by utilizing multiple CPU cores effectively. The program also prints which thread is working on which index, allowing us to see how work is distributed across threads.

        ### **Key Points in Parallel Code**:
        - `#pragma omp parallel`: This directive is used to start parallel execution. Threads are created here.
        - `#pragma omp for`: This distributes the loop iterations among the available threads.
        - `omp_get_thread_num()`: This function returns the thread ID, allowing us to print which thread is working on a particular iteration.

        In the OpenMP code, the work of adding the arrays is done in parallel, making it faster for larger data sets. However, the correctness of the result remains the same.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Serial code
    serial_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>

    int main (int argc, char *argv[]) {
        const int N = 20;
        int i;
        double a[N], b[N], result[N];

        // Initialize arrays
        for (i = 0; i < N; i++) {
            a[i] = 1.0 * i;
            b[i] = 2.0 * i;
        }

        // Perform element-wise addition
        for (i = 0; i < N; i++) {
            result[i] = a[i] + b[i];
        }

        // Print test result
        printf(\"TEST result[19] = %g\\n\", result[19]);

        return 0;
    }
    \"\"\"

    # Save the serial code to a file
    with open(\"serial_code.c\", \"w\") as f:
        f.write(serial_code)

    # Compile the serial program
    !gcc serial_code.c -o serial_code

    # Run the serial program
    !./serial_code
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # OpenMP parallel code
    openmp_code = \"\"\"
    #include <omp.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main (int argc, char *argv[]) {
        const int N = 20;
        int i;
        double a[N], b[N], result[N];

        // Initialize arrays
        for (i = 0; i < N; i++) {
            a[i] = 1.0 * i;
            b[i] = 2.0 * i;
        }

        // Parallel region begins here
        #pragma omp parallel private(i)
        {
            int threadid = omp_get_thread_num();  // Get thread id

            // Parallel loop: each thread processes part of the arrays
            #pragma omp for
            for (i = 0; i < N; i++) {
                result[i] = a[i] + b[i];
                printf(\"Thread id: %d working on index %d\\n\", threadid, i);
            }
        } // Parallel region ends

        // Print test result
        printf(\"TEST result[19] = %g\\n\", result[19]);

        return 0;
    }
    \"\"\"

    # Save the OpenMP code to a file
    with open(\"openmp_code.c\", \"w\") as f:
        f.write(openmp_code)

    # Compile the parallel program with OpenMP support
    !gcc -fopenmp openmp_code.c -o openmp_code

    # Run the OpenMP program
    !./openmp_code
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Same codes but measuring the time.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Serial code with time measurement using clock()
    serial_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>  // For measuring time using clock()

    int main (int argc, char *argv[]) {
        const int N = 300000;  // Reduce the size for testing purposes
        int i;
        double a[N], b[N], result[N];
        clock_t start_time, end_time;

        // Initialize arrays
        for (i = 0; i < N; i++) {
            a[i] = 1.0 * i;
            b[i] = 2.0 * i;
        }

        // Start measuring time
        start_time = clock();

        // Perform element-wise addition
        for (i = 0; i < N; i++) {
            result[i] = a[i] + b[i];
        }

        // End measuring time
        end_time = clock();

        // Print test result and time taken
        printf(\"TEST result[N-1] = %g\\n\", result[N-1]);
        fflush(stdout);  // Ensure immediate output
        printf(\"Time taken by serial code: %f seconds\\n\", (double)(end_time - start_time) / CLOCKS_PER_SEC);
        fflush(stdout);  // Ensure immediate output

        return 0;
    }
    \"\"\"

    # Save the serial code to a file
    with open(\"serial_code.c\", \"w\") as f:
        f.write(serial_code)

    # Compile the serial program
    !gcc serial_code.c -o serial_code

    # Run the serial program
    !./serial_code
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # OpenMP parallel code with time measurement
    openmp_code = \"\"\"
    #include <omp.h>
    #include <stdio.h>
    #include <stdlib.h>

    int main (int argc, char *argv[]) {
        const int N = 300000;  // Reduce the size for testing purposes
        int i;
        double a[N], b[N], result[N];
        double start_time, end_time;

        // Initialize arrays
        for (i = 0; i < N; i++) {
            a[i] = 1.0 * i;
            b[i] = 2.0 * i;
        }

        // Start measuring time
        start_time = omp_get_wtime();

        // Parallel region begins here
        #pragma omp parallel private(i)
        {
            int threadid = omp_get_thread_num();  // Get thread id

            // Parallel loop: each thread processes part of the arrays
            #pragma omp for
            for (i = 0; i < N; i++) {
                result[i] = a[i] + b[i];
            }
        } // Parallel region ends

        // End measuring time
        end_time = omp_get_wtime();

        // Print test result and time taken
        printf(\"TEST result[N-1] = %g\\n\", result[N-1]);
        fflush(stdout);  // Ensure immediate output
        printf(\"Time taken by OpenMP code: %f seconds\\n\", end_time - start_time);
        fflush(stdout);  // Ensure immediate output

        return 0;
    }
    \"\"\"

    # Save the OpenMP code to a file
    with open(\"openmp_code.c\", \"w\") as f:
        f.write(openmp_code)

    # Compile the parallel program with OpenMP support
    !gcc -fopenmp openmp_code.c -o openmp_code

    # Run the OpenMP program
    !./openmp_code
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Modifying OpenMP Code with Time Measurement

        In this exercise, you will modify the existing OpenMP code to better understand how time measurement and thread management work in OpenMP.

        ### Task: Measure Time for Each Thread
        The current program measures the total execution time of the parallel region, but it doesn't give any information about the time each thread takes to complete its work. Modify the program so that:
        1. Each thread measures its own execution time within the parallel region.
        2. Print the execution time for each thread after the parallel loop.

        ### Hint:
        - Use `omp_get_wtime()` within the parallel region to measure the time at the start and end of the thread's execution.
        - You can print the thread ID and its execution time inside the parallel block after the loop.

        After making these changes, run the program and observe how the execution time varies between threads.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise 7.2: OpenMP Directives and Clauses

        This exercise will guide you through the use of OpenMP directives and clauses, focusing on parallel regions, controlling the number of threads, and data sharing among threads.

        ## 7.2.1 Parallel Regions
        A parallel region in OpenMP is a block of code that runs simultaneously across multiple threads. This is initiated using the `#pragma omp parallel` directive.

        ### 7.2.1.1 num_threads Clause
        The `num_threads` clause specifies the exact number of threads to be used in the parallel region. This is important for optimizing performance and ensuring proper resource utilization.

        ### 7.2.1.2 default Clause
        The `default` clause specifies the default data-sharing attributes for variables within a parallel region. It can be set to `shared`, `private`, or `none`, determining how variables are accessed by threads.

        ## 7.2.2 Assigning the Number of Threads
        Assigning the number of threads can be done inside the code using the `num_threads` clause or outside the code using environment variables. Both methods have their own use cases and advantages.

        ## 7.2.3 Work-sharing Constructs
        Work-sharing constructs in OpenMP, like `#pragma omp for` and `#pragma omp sections`, are used to divide tasks among threads, allowing for efficient parallel execution.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Example of using OpenMP directives and clauses
    # Save this C code to a file named \"omp_directives.c\"

    omp_code = \"\"\"
    #include <omp.h>
    #include <stdio.h>

    int main() {
        #pragma omp parallel num_threads(4)
        {
            int id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            if (id == 0) {
                printf(\"Total number of threads: %d\\n\", num_threads);
            }
            printf(\"Thread %d is running\\n\", id);
        }
        return 0;
    }
    \"\"\"

    # Write the OpenMP code to a file
    with open('omp_directives.c', 'w') as f:
        f.write(omp_code)

    # Compile the OpenMP C code using GCC with the -fopenmp flag
    !gcc -fopenmp -o omp_directives omp_directives.c

    # Run the compiled program
    !./omp_directives
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OpenMP Program with Parallel Sections

        In this example, we will explore how OpenMP can be used to parallelize different tasks using sections. This code performs the following tasks:
        1. Initializes an array `x[]` with values from 0 to 99.
        2. Divides the array into two groups based on a `divide` threshold.
        3. Calculates the sum of the elements in the array.
        4. Calculates the sum of the squares of the elements in the array.

        Each of these tasks is done in parallel using OpenMP sections. This is an example of **task parallelism** where different threads work on different parts of the computation simultaneously.

        ### **Key OpenMP Concepts Used**
        - **`#pragma omp parallel for`**: This directive is used to parallelize the initialization of the array `x[]`. Each thread will initialize a portion of the array.
        - **`#pragma omp sections`**: This is used to split the program into different sections, where each section is executed by a separate thread.
        - **Shared and Private Variables**: The variable `x[]` is shared between threads, while `i` is private for each thread, meaning that each thread has its own copy of `i`.

        ### **Code Explanation**
        - The array `x[]` is first initialized using a parallel loop.
        - The program then forks into different threads using `#pragma omp sections`. Each thread works on a different section:
            - One thread counts the number of values in `x[]` that are below or above a given threshold (`divide`).
            - Another thread calculates the sum of all the elements in `x[]`.
            - A third thread calculates the sum of the squares of the elements in `x[]`.

        By parallelizing these tasks, we can speed up the program and utilize multiple cores effectively.

        ### **C Code with OpenMP Sections**

        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 1: Writing the C code to a file
    openmp_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>
    #include <omp.h>

    int main() {
        const int N = 100;
        int x[N], i, sum, sum2;
        int upper = 0, lower = 0;  // Initialize upper and lower
        int divide = 20;
        sum = 0;
        sum2 = 0;

        // Parallelize the initialization of the array
        #pragma omp parallel for
        for (i = 0; i < N; i++) {
            x[i] = i;
        }

        // Parallel region with sections
        #pragma omp parallel private(i) shared(x, upper, lower)
        {
            // Fork several different threads using sections
            #pragma omp sections
            {
                // First section: Count elements below and above the divide threshold
                #pragma omp section
                {
                    for (i = 0; i < N; i++) {
                        if (x[i] > divide) upper++;
                        if (x[i] <= divide) lower++;
                    }
                    printf(\"The number of points at or below %d in x is %d\\n\", divide, lower);
                    printf(\"The number of points above %d in x is %d\\n\", divide, upper);
                }

                // Second section: Calculate the sum of elements in x
                #pragma omp section
                {
                    for (i = 0; i < N; i++) {
                        sum = sum + x[i];
                    }
                    printf(\"Sum of x = %d\\n\", sum);
                }

                // Third section: Calculate the sum of squares of elements in x
                #pragma omp section
                {
                    for (i = 0; i < N; i++) {
                        sum2 = sum2 + x[i] * x[i];
                    }
                    printf(\"Sum2 of x = %d\\n\", sum2);
                }
            }
        }

        return 0;
    }
    \"\"\"

    # Step 2: Save the C code to a file
    with open(\"openmp_code.c\", \"w\") as file:
        file.write(openmp_code)

    # Step 3: Compile the C code with OpenMP support
    !gcc -fopenmp openmp_code.c -o openmp_code

    # Step 4: Run the compiled program
    !./openmp_code
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Modifying OpenMP Sections with Shared and Private Variables

        In this exercise, you will modify the existing OpenMP code to better understand how shared and private variables work in OpenMP, especially when using sections.

        ### Task: Fix the Issue with Shared Variables
        Currently, the variables `upper` and `lower` are shared among all threads. However, multiple threads are trying to modify these shared variables simultaneously, which can cause incorrect results (race conditions). Modify the program so that:
        1. Each thread has its own private copy of `upper` and `lower`.
        2. After each thread finishes its work, the results should be combined into the shared `upper` and `lower` variables in a safe manner.

        ### Hint:
        - Use the `private` clause to make `upper` and `lower` private within the sections.
        - Use the `reduction` clause or a critical section to safely combine the results from each thread.

        After making these changes, run the program and check whether the results for `upper` and `lower` are correct.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise 7.3: Data Environment in OpenMP

        This exercise explores the data-sharing clauses in OpenMP, including `shared`, `private`, `firstprivate`, and `reduction`, which control how variables are accessed and modified within parallel regions.

        ## 7.3.1 Data Sharing Clauses
        ### 7.3.1.1 Shared Clause
        The `shared` clause makes a variable accessible to all threads in a parallel region, which can lead to race conditions if not synchronized properly.

        ### 7.3.1.2 Private Clause
        The `private` clause ensures that each thread has its own instance of a variable, which is useful for thread-specific computations.

        ### 7.3.1.3 Firstprivate Clause
        The `firstprivate` clause initializes private variables with the value from the master thread, ensuring consistent initial states across threads.

        ## 7.3.2 Reduction Clause
        The `reduction` clause is used to perform a reduction operation (e.g., sum, product) on variables across all threads, combining their results into a single value.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Example of using data-sharing clauses in OpenMP
    # Save this C code to a file named \"omp_data_clauses.c\"

    omp_data_code = \"\"\"
    #include <omp.h>
    #include <stdio.h>

    int main() {
        int n = 1000;
        int sum = 0;
        int array[1000];

        // Initialize the array
        for (int i = 0; i < n; i++) {
            array[i] = i + 1;
        }

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < n; i++) {
            sum += array[i];
        }

        printf(\"Total Sum: %d\\n\", sum); // Correct total sum
        return 0;
    }
    \"\"\"

    # Write the OpenMP code to a file
    with open('omp_data_clauses.c', 'w') as f:
        f.write(omp_data_code)

    # Compile the OpenMP C code using GCC with the -fopenmp flag
    !gcc -fopenmp -o omp_data_clauses omp_data_clauses.c

    # Run the compiled program
    !./omp_data_clauses
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Understanding Data-Sharing Clauses in OpenMP

        In this exercise, you will modify the existing OpenMP code to explore the behavior of data-sharing clauses in OpenMP, particularly focusing on shared and private variables.

        ### Task: Modify the Code to Make the Array Private
        The current program calculates the sum of an array using OpenMP with the `reduction` clause. However, the array `array[]` is shared across all threads by default. Modify the program so that:
        1. Each thread has its own private copy of the array during the parallel loop.
        2. After the parallel loop, ensure that the final result is still correct.

        ### Hint:
        - You can use the `private` or `firstprivate` clause to make the array private to each thread.
        - Ensure that the initialization of the array happens outside the parallel region, or make sure the array is initialized properly in each thread if you use `firstprivate`.

        Run the program after making these changes and compare the results. This will help you understand how data-sharing clauses work in OpenMP.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise 7.4: Synchronization Techniques in OpenMP

        In this exercise, you will learn about synchronization techniques in OpenMP, including critical sections, atomic operations, barriers, and locks.

        ## 7.4.1 Critical Sections
        A critical section is a block of code that must be executed by only one thread at a time, ensuring mutual exclusion.

        ## 7.4.2 Atomic Operations
        Atomic operations provide a lightweight synchronization mechanism for simple updates to shared variables.

        ## 7.4.3 Barrier Synchronization
        The `#pragma omp barrier` directive ensures that all threads reach a specific point before any can proceed, useful for coordinating tasks.

        ## 7.4.4 Locks
        Locks provide fine-grained control over access to critical sections, allowing threads to acquire and release locks manually.

        """
    )
    return


app._unparsable_cell(
    r"""
    # Example of synchronization techniques in OpenMP
    # Save this C code to a file named \"omp_sync.c\"

    omp_sync_code = \"\"\"
    #include <omp.h>
    #include <stdio.h>

    int main() {
        int balance = 0;
        omp_lock_t lock;

        omp_init_lock(&lock);

        #pragma omp parallel for
        for (int i = 0; i < 1000; i++) {
            omp_set_lock(&lock);
            balance += 1;
            omp_unset_lock(&lock);
        }

        omp_destroy_lock(&lock);

        printf(\"Final Balance: %d\\n\", balance);
        return 0;
    }
    \"\"\"

    # Write the OpenMP code to a file
    with open('omp_sync.c', 'w') as f:
        f.write(omp_sync_code)

    # Compile the OpenMP C code using GCC with the -fopenmp flag
    !gcc -fopenmp -o omp_sync omp_sync.c

    # Run the compiled program
    !./omp_sync
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Understanding Synchronization Techniques in OpenMP

        In this exercise, you will modify an existing OpenMP program to explore how synchronization works in OpenMP. The program currently uses locks (`omp_set_lock()` and `omp_unset_lock()`) to ensure that multiple threads do not modify the `balance` variable simultaneously.

        ### Task: Replace Locks with Critical Sections
        In the current program, locks are used to ensure that the `balance` is updated correctly. Modify the program so that:
        1. You remove the lock mechanism.
        2. Instead of using locks, use the `#pragma omp critical` directive to ensure that only one thread at a time modifies the `balance`.

        ### Hint:
        - Use the `#pragma omp critical` directive in place of the lock around the `balance` update.
        - Test the program and check if the final balance remains correct.

        ### Task 2: Experiment with Race Conditions (Optional)
        Remove the `critical` directive entirely and observe the effect of the race condition when multiple threads try to modify the `balance` without any synchronization. Discuss why the result is incorrect.

        Run the modified program and observe the results. This will help you understand the importance of synchronization techniques like locks and critical sections in OpenMP.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise: Parallelizing Code with OpenMP

        In this exercise, you will learn how to parallelize a code step by step using OpenMP. The goal is to transform a serial program into a parallel program by applying OpenMP directives.

        You are given a serial code that performs the following tasks:
        1. Initializes an array with values.
        2. Computes the sum of all elements in the array.
        3. Computes the sum of squares of all elements in the array.

        ### **Steps to Parallelize the Code**

        #### **Step 1: Analyze the Serial Code**
        First, we provide you with the serial version of the code. Review it carefully. Your task is to parallelize the initialization, summation, and sum of squares calculation using OpenMP.

        ```c
        #include <stdio.h>
        #include <stdlib.h>

        int main() {
            const int N = 1000000;  // Size of the array
            int i;
            double x[N], sum = 0.0, sum2 = 0.0;

            // Step 1: Initialize the array
            for (i = 0; i < N; i++) {
                x[i] = i * 1.0;
            }

            // Step 2: Calculate the sum of all elements
            for (i = 0; i < N; i++) {
                sum += x[i];
            }

            // Step 3: Calculate the sum of the squares of all elements
            for (i = 0; i < N; i++) {
                sum2 += x[i] * x[i];
            }

            printf("Sum: %f\n", sum);
            printf("Sum of squares: %f\n", sum2);

            return 0;
        }


        ###Step 2: Add OpenMP Directives
        Now, try to parallelize the following parts of the code using OpenMP:

        Initialization of the array: Use the #pragma omp parallel for directive to parallelize the loop that initializes the array.
        Summing the elements: Add the #pragma omp parallel for directive to parallelize the summation loop.
        Summing the squares: Again, use #pragma omp parallel for to parallelize the loop that computes the sum of squares.
        Keep in mind:

        You need to ensure thread safety when multiple threads are updating shared variables (sum, sum2). You can achieve this by using the OpenMP reduction clause.

        Step 3: Check for Correctness
        Once you've parallelized the code, run it and check whether the output matches the serial version. Make sure that the final sum and sum of squares are correct.

        After you've attempted the exercise, we'll provide the solution.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 2: Writing the parallelized C code with OpenMP to a file
    parallel_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>
    #include <omp.h>

    int main() {

    # Add your code here


        return 0;
    }
    \"\"\"

    # Save the parallel code to a file
    with open(\"parallel_code.c\", \"w\") as file:
        file.write(parallel_code)

    # Compile the parallel code with OpenMP support
    !gcc -fopenmp parallel_code.c -o parallel_code

    # Run the compiled parallel program
    !./parallel_code
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""




        ###Solution to the exercise
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Serial Code so you can run it:
        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 1: Writing the serial version of the C code to a file (for students to parallelize)
    serial_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>

    int main() {
        const int N = 1000000;  // Size of the array
        int i;
        double x[N], sum = 0.0, sum2 = 0.0;

        // Step 1: Initialize the array
        for (i = 0; i < N; i++) {
            x[i] = i * 1.0;
        }

        // Step 2: Calculate the sum of all elements
        for (i = 0; i < N; i++) {
            sum += x[i];
        }

        // Step 3: Calculate the sum of the squares of all elements
        for (i = 0; i < N; i++) {
            sum2 += x[i] * x[i];
        }

        printf(\"Sum: %f\\n\", sum);
        printf(\"Sum of squares: %f\\n\", sum2);

        return 0;
    }
    \"\"\"

    # Save the serial code to a file
    with open(\"serial_code.c\", \"w\") as file:
        file.write(serial_code)

    # Compile the serial code (no OpenMP needed here)
    !gcc serial_code.c -o serial_code

    # Run the serial program
    !./serial_code
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        and the solutions with the parallel code
        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 2: Writing the parallelized C code with OpenMP to a file
    parallel_code = \"\"\"
    #include <stdio.h>
    #include <stdlib.h>
    #include <omp.h>

    int main() {
        const int N = 1000000;  // Size of the array
        int i;
        double x[N], sum = 0.0, sum2 = 0.0;

        // Step 1: Parallelize array initialization
        #pragma omp parallel for
        for (i = 0; i < N; i++) {
            x[i] = i * 1.0;
        }

        // Step 2: Parallelize the summation with reduction
        #pragma omp parallel for reduction(+:sum)
        for (i = 0; i < N; i++) {
            sum += x[i];
        }

        // Step 3: Parallelize the sum of squares with reduction
        #pragma omp parallel for reduction(+:sum2)
        for (i = 0; i < N; i++) {
            sum2 += x[i] * x[i];
        }

        printf(\"Sum: %f\\n\", sum);
        printf(\"Sum of squares: %f\\n\", sum2);

        return 0;
    }
    \"\"\"

    # Save the parallel code to a file
    with open(\"parallel_code.c\", \"w\") as file:
        file.write(parallel_code)

    # Compile the parallel code with OpenMP support
    !gcc -fopenmp parallel_code.c -o parallel_code

    # Run the compiled parallel program
    !./parallel_code
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ####Explanation of the Solution
        Parallelizing the Initialization: The #pragma omp parallel for directive is used to distribute the loop iterations among threads. Each thread will initialize a portion of the array.

        Parallelizing the Sum Calculation: Since multiple threads will be updating the sum variable, we use the reduction(+:sum) clause. This ensures that each thread will maintain a local copy of sum, and at the end of the parallel region, these local sums will be combined.

        Parallelizing the Sum of Squares: Similar to the sum calculation, the reduction(+:sum2) clause is used to handle the concurrent updates to the sum2 variable.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

