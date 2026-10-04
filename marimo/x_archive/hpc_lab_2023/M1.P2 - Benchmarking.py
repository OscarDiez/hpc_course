import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # M1.P2. Benchmarking
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## I. Why Benchmark?
        In this course we will understand about HPC, parallelism, concurrency, and distributed computing. By the end of the course you should be able to  modify serial code to make it parallel or concurrent, where appropriate, and you should have some concepts on writing parallel algorithms and workflows yourself. In order to do this well, you need to be able to benchmark your algorithms and the systems they run on. You know the basics of how to analyze parallel algorithms, but it's always useful to collect empirical measurements, and it's also usually easier to collect empirical data than it is to do mathematical analysis of parallel algorithms.

        So, now that we're through with that, what is benchmarking? A _benchmark_ is the act of running a computer program, a set of programs, or other operations, in order to assess the relative performance of an object, normally by running a number of standard tests and trials against it. Benchmarking is usually associated with assessing performance characteristics of computer hardware, for example, the floating point operation performance of a CPU, but there are circumstances when the technique is also applicable to software. In this notebook, we will be doing a bit of both. The following graph is an example of historical benchmarks going back to 1996:
        ![benchmarks of old](http://preshing.com/images/float-point-perf.png)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2.1 - Reading Files Vs Reading From Memory
        When possible, you should try and stay away from reading and writing to disk as much as possible? It is much slower to read and write to disk than it is to memory. In this example, we're going to quantify that. I've designed a [python decorator](https://www.python.org/dev/peps/pep-0318/) to be used as a timer. In order to use it, you "decorate" a function with it by putting `@timed` over the function definition. You'll see what I mean by this in the code cell below. In this example, we're going to create a multidimensional array, store it to memory, and read it from memory. Then, we're going to do the same with a file. We'll use the decorator to time it.
        """
    )
    return


@app.cell
def _():
    import time

    def timed(func):
        """
          decorator to calculate the total
          time of a function
        """

        def st_func(*args, **keyArgs):
            t1 = time.time()
            r = func(*args, **keyArgs)
            t2 = time.time()
            print( "Function=%s, Time=%ssec" % (func.__name__, t2 - t1))
            return r

        return st_func

    @timed
    def memory_test():
        import random
        arr = [[[x for x in range(100)] for y in range(100)] for z in range(10)]
        print(len(arr))
    
    @timed
    def file_test():
        import ast
        # Note that the array is much smaller in the file test
        # But it is still **MUCH** slower
        arr = [[[x for x in range(100)] for y in range(100)] for z in range(10)]
        with open("data/matrix.out", "w") as f:
            f.write(str(arr))
        with open("data/matrix.out") as f:
            arr = ast.literal_eval(f.read())
        print(len(arr))
    return file_test, memory_test, time


@app.cell
def _(file_test, memory_test):
    memory_test()
    file_test()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## II. Single Machine Benchmarking
        What we did in the last example was a benchmark of a single core algorithm, in order to learn for sure that file IO is much slower than reading and writing to memory. Now, we want to test a parallel algorithm and see how fast we can get it to go, on a single machine. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2.2 - Timing Parallel Algorithms on One Machine
        To time parallel algorithms, we're going to use our monte carlo frog simulation from before. We'll time it and graph the times for different numbers of cores. Recall that there were three for loops nested within each other and we parallelized the outer loop. As an exercise, feel free to parallelize the middle and inner loops.
        """
    )
    return


@app.cell
def _(random, time):
    import math
    from multiprocessing import Pool

    def outer_loop(numJumps):
        numTries = 1000
        lenJump = 1
        numSuccesses = 0
        for j in range(numTries):
            frogPosition = [0.0, 0.0]
            for _i in range(0, numJumps, 1):
                theta = random.uniform(0, 2 * math.pi)
                frogPosition[0] += lenJump * math.cos(theta)
                frogPosition[1] += lenJump * math.sin(theta)
            frogMagnitude = (frogPosition[0] ** 2 + frogPosition[1] ** 2) ** 0.5
            if frogMagnitude <= 1.0:
                numSuccesses += float(1)
        successRate = float(numSuccesses / numTries)
    times = []
    for _i in range(1, 32, 2):
        pool = Pool(_i)
        _before = time.time()
        tasks = [pool.apply_async(outer_loop, (j,)) for j in range(1, 20)]
        tasks[-1].get()
        _after = time.time()
        _tot = _after - _before
        times.append(_tot)
        print('{} Cores, {} Sec'.format(_i, _tot))
    return (times,)


@app.cell
def _(times):
    # '%matplotlib inline' command supported automatically in marimo
    # Plot the times we collected
    import matplotlib.pyplot as plt
    import numpy as np

    #h Helper function to plot an equation
    def graph(formula, x_range):  
        x = np.array(x_range)  
        y = formula(x) 
        plt.plot(x, y)  

    # Graph theoretical maximum (in blue)
    graph(lambda x: 23/x, range(1, 32))

    # Graph empirical data (in orange)
    plt.plot(range(1,32, 2), times)

    plt.ylabel("Execution Time (sec)")
    plt.xlabel("Number of cores")
    plt.title("Execution Time vs Number of Cores for Monte Carlo Frog Simulation")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## III. Distributed Benchmarking
        Distributed benchmarking is, as it sounds, benchmarking a distributed algorithm on a distributed system. Many of the considerations you need to make when you write a distributed algorithm also need to be made when you benchmark a distributed algorithm as well. You need to worry about what parts of your algorithm can run concurrently, as often, things can be offloaded to remote machines if they can be run concurrently. You need to worry about which parts of your process need to access things from other parts, because you can't depend on all of the remote parts of your code having access to the same memory pool as any other part. Because of this, you need to worry about how you can have the processes communicate with each other. Because of all of this complexity, you need to make sure you know how you can ensure that your benchmarking does not affect the output or speed of your algorithm.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## System Benchmarking
        So we've done a bunch of benchmarking of our software. Next, we are going to do some hardware benchmarking. Earlier in the notebook, I said "Benchmarking is usually associated with assessing performance characteristics of computer hardware, for example, the floating point operation performance of a CPU", and we're about to calculate exactly that. 

        The way we will do this is by the following process: First, come up with a task that will take a constant amount of computer power. Then, compute what that amount of power is, by counting the number of floating point instructions that it will take to run the code. Then, repeatedly time the code and divide the number of floating point instructions by the time it took to get a measurement in FLOPS of how performant our computer is.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2.4 - Using a Workflow to Estimate System Power
        We're going to perform a fixed number of floating point instructions, specifically floating point adds, in a Python loop. Note that there is significant overhead involved with starting a python loop, so the quote of roughly 20 MFLOPS (last time I tested it) is quite a low estimate for this single machine.
        """
    )
    return


@app.cell
def _(time):
    _before = time.time()
    for _i in range(10 ** 6):
        floating_point = 1.0
        float_increment = 1.0
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
        floating_point += float_increment
    _after = time.time()
    _tot = _after - _before
    print('20M FLOP in {} sec, {}GFLOPS'.format(_tot, 20.5 ** 7 / _tot / 10 ** 6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##  End of practice

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

