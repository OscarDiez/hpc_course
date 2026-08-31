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


app._unparsable_cell(
    r"""
    # Simple example of parallelizing a loop with OpenMP
    # Save this C code to a file named \"simple_openmp.c\"

    hello_openmp = \"\"\"
    #include <stdio.h>
    #include <omp.h>

    int main() {
        // Set the number of threads, you can change the number here
        omp_set_num_threads(4);  // For example, setting 4 threads

        // Start parallel region
        #pragma omp parallel
        {
            // Get the thread number
            int thread_id = omp_get_thread_num();

            // Get the total number of threads
            int total_threads = omp_get_num_threads();

            // Each thread prints its ID and the total number of threads
            printf(\"Hello World from thread %d out of %d threads\\n\", thread_id, total_threads);
        }

        return 0;
    }
    \"\"\"

    # Write the OpenMP code to a file
    with open('hello_openmp.c', 'w') as f:
        f.write(hello_openmp)

    # Compile the OpenMP C code using GCC with the -fopenmp flag
    !gcc -fopenmp -o hello_openmp hello_openmp.c

    # Run the compiled program
    !./hello_openmp
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Simple example of parallelizing a loop with OpenMP
    # Save this C code to a file named \"simple_openmp.c\"

    simple_openmp = \"\"\"
    #include <omp.h>
    #include <stdio.h>

    void process_array(float *array, int size) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            array[i] = array[i] * 2.0;
        }
    }

    int main() {
        int size = 1000;
        float array[size];

        // Initialize the array
        for (int i = 0; i < size; i++) {
            array[i] = i * 1.0;
        }

        process_array(array, size);

        // Print the first 10 elements to verify
        for (int i = 0; i < 10; i++) {
            printf(\"array[%d] = %f\\n\", i, array[i]);
        }

        return 0;
    }
    \"\"\"

    # Write the OpenMP code to a file
    with open('simple_openmp.c', 'w') as f:
        f.write(simple_openmp)

    # Compile the OpenMP C code using GCC with the -fopenmp flag
    !gcc -fopenmp -o simple_openmp simple_openmp.c

    # Run the compiled program
    !./simple_openmp
    """,
    name="_"
)


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


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

