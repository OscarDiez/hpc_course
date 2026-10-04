import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Testing OpenMP in C using Google Colab

        This notebook will compile and run a simple C program using OpenMP to demonstrate parallel processing.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Step 1: Install the GCC compiler with OpenMP support
    !apt-get update && apt-get install -y gcc

    # Step 2: Write the C program to a file
    c_program = '''
    #include <omp.h>
    #include <stdio.h>

    int main(int argc, char *argv[]) {

     int nthreads, tid;

     /* Fork a team of threads with each thread having a private tid variable */
     #pragma omp parallel private(tid)
       {

       /* Obtain and print thread id */
       tid = omp_get_thread_num();
       printf(\"Hello World! from thread = %d\n\", tid);

       /* Only master thread does this */
       if (tid == 0) 
         {
         nthreads = omp_get_num_threads();
         printf(\"Number of threads = %d\n\", nthreads);
         }

       }  /* All threads join master thread and terminate */

     }
    '''

    with open(\"omp_hello.c\", \"w\") as file:
        file.write(c_program)

    print(\"C program written to omp_hello.c\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Step 3: Compile the C program using gcc with OpenMP support
    !gcc -o omp_hello -fopenmp omp_hello.c

    # Step 4: Run the compiled program
    !./omp_hello
    """,
    name="_"
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

