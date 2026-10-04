import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Ch. 6. Algorithm Analysis 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## I. Intro to Parallel Algorithm Analysis
        This course assumes you have done some computer science, but does not assume any knowledge of algorithm design or formal algorithm analysis. However, I still want to do some basic analysis of some parallel algorithms and serial algorithms, so you have some understanding of what kinds of things will make your code faster, and what kind of code can even be made faster. We're going to be very informal about our algorithm analysis because it's more fun that way, and we will still get a few important points across. After this chapter, I want you to have some intuition that helps you understand why this graph looks the way it does:
        ![parallel time graph](https://ars.els-cdn.com/content/image/1-s2.0-S0098300413001465-gr9.jpg)
        So, let's get started. We talked, as far back as Ch. 3, about what a parallel algorithm looks like and what makes an algorithm parallel. We talked about trivially parallelizable algorithms and monte carlo simulations. What we didn't talk about, specifically, is how these different algorithms change how workflows run. Essentially, if something is trivially parallelizable, if you go from 1 core to _n_ cores, without changing any parameters, you should see a roughly _n_ times speedup. This means that the run time of this program should be roughly $\frac{1}{n}\times (original\ running\ time)$. This means, as you add more cores to your trivially parallelizable code, you should see your code's runtime shrink as $\frac{1}{n}$. This graph is what that looks like:
        ![1/n](https://i.imgur.com/MbqqIAn.png)

        This graph assumes that your code initally took 10 seconds to run, and by the time you add five cores, you already only take ten seconds to run your code.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 6.1 - Monte Carlo Analysis
        Given what we just talked about, Monte Carlo simulations should fit nicely into that class of algorithms. We should see a linear speedup and a resulting runtime proportional to $\frac{1}{n}$. In this example, we're going to investigate that and try to show with empirical data that this holds for this case. This should hopefully complement our theoretical analysis from earlier and help convince you that this makes sense. We're going to use our parallel monte carlo pi from earlier, and time it as we scale it.
        """
    )
    return


@app.cell
def _():
    import random
    import multiprocessing
    from multiprocessing import Pool

    def monte_carlo_pi_part(n):
        _count = 0
        for _i in range(_n):
            x = random.random()
            y = random.random()
            if x * x + y * y <= 1:
                _count = _count + 1
        return _count
    return Pool, monte_carlo_pi_part, multiprocessing, random


@app.cell
def _(Pool, monte_carlo_pi_part):
    import time
    _n = 1000000
    times = []
    for _np in range(1, 32, 2):
        start = time.time()
        _part_count = [_n // _np for _i in range(_np)]
        _pool = Pool(processes=_np)
        _count = _pool.map(monte_carlo_pi_part, _part_count)
        pi = sum(_count) / (_n * 1.0) * 4
        end = time.time()
        _tot = end - start
        times.append(_tot)
        print('Esitmated value of Pi: {} ({} cores, time: {}s)'.format(pi, _np, _tot))
    return time, times


@app.cell
def _(times):
    import matplotlib.pyplot as plt
    import numpy as np

    def graph(formula, x_range):
        x = _np.array(x_range)
        y = formula(x)
        plt.plot(x, y)
    graph(lambda x: 32 / x, range(1, 32))
    plt.plot([_i * 2 for _i in range(1, 17)], times)
    plt.ylabel('Execution Time (sec)')
    plt.xlabel('Number of cores')
    plt.title('Execution Time vs Number of Cores for Monte Carlo Pi')
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## II. Strictly Serial Algorithms
        We've seen how algorithms that respond very well to scale do, now what about the other end of the spectrum? If there are algorithms that can be parallelized really easily, it seems like there _should_ be some that can not. Why? Well, it just feels like it! While that's not really valid logic, the assumption is correct. There's a family of algorithms that can simply not be parallelized. This family is called _strictly serial algorithms_, and they are more common than you might think. One such common example is the Fibonacci sequence. The Fibonacci sequence is defined recursively, such that the _n_-th number in the sequence is the sum of the _n-1_-th and the _n-2_-th. That is to say that _fib(n)_ = _fib(n-1)+fib(n-2)_. Because of this, dependence on the previous values, computing the _n_-th Fibonacci number is a strictly serial task. (This is not actually true, as the Fibonacci sequence does indeed have a closed form solution, but that's not the point of this exercise. There are sequences which only have recursive forms, but they're more complicated). This image shows how Fibonacci number computations are always recursively broken down into calls of `fib(1)` or `fib(o)`:
        ![fib call tree](https://i.stack.imgur.com/7iU1j.png)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 6.2 - Fibonacci Sequence Generator
        Because we mentioned how the Fibonacci sequence is a strictly serial algorithm, we're now going to use it as an example. We will write a recursive Fibonacci sequence generator and perform some analysis.
        """
    )
    return


@app.function
def fib(n):
    if _n == 0 or _n == 1:
        return 1
    return fib(_n - 1) + fib(_n - 2)


@app.cell
def _(time):
    times_1 = []
    for _i in range(0, 40, 2):
        before = time.time()
        res = fib(_i)
        after = time.time()
        _tot = after - before
        print('fib({}) = {} ({}sec)'.format(_i, res, _tot))
        times_1.append(_tot)
    return (times_1,)


@app.cell
def _(plt, times_1):
    plt.plot([2 * _i for _i in range(20)], times_1)
    plt.ylabel('Execution Time (sec)')
    plt.xlabel('Number in Fibonacci sequence')
    plt.title('Execution Time vs Number in Fibonacci Sequence')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Hopefully, you can see that attempting to parallelize any computation like this is hopeless. 

        Let's examine the call tree for calculating the fifth fibonacci number:
        ```
        fib(5) -> fib(4)+fib(3)
                    |      |
              fib(3)+fib(2)|
                        fib(2)+fib(1)
        and so on
        ```

        Now, the execution will still be sequential. This is because, even if you could fork a thread for each call in the tree, it would spawn so many new threads that it simply wouldn't be worth it to do that. It would slow the program down so much that it would end up slower than the serial algorithm you started with.  

        In general, you want to parallelize only "embarrassingly parallel" tasks. That is, tasks which are computationally expensive and can be computed independently. Many people forget about the first part. Threads are so expensive that it only makes sense to make one when you have a huge amount of work for them to do, and moreover, that you can devote an entire processor to the thread. If you have 8 processors then making 80 threads is going to force them to share the processor, slowing each one of them down tremendously. You'd do better to make only 8 threads and let each have 100% access to the processor when you have an embarrassingly parallel task to perform.

        In this case, it is theoretically possible to parallelize the algorithm (slightly), but practically not worth it. In some cases, such as non primitive recursive problems (like [the Ackermann function](https://en.wikipedia.org/wiki/Ackermann_function)), it's completely impossible. 

        Also, note that this could be sped up by a lot through the use of [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming), but that's not what this course is about.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## III. Task Parallelism vs. Data Parallelism
        As you may gather from the title of this section, there are two flavors of parallelism which are important: task parallelism and data parallelism. Do you remember when we talked about Single Instruction, Multiple Data (SIMD)? Well, if you did (or if you didn't), you might be able to guess that SIMD is an example of data parallelism. Let's define data parallelism better. 

        Data parallelism focuses on distributing the data across different nodes, which operate on the data in parallel. It can be applied on regular data structures like arrays and matrices by working on each element in parallel. It contrasts to task parallelism as another form of parallelism, where tasks are distributed rather than data (we'll get to this in a moment). 

        As this definition suggests, data parallelism occurs when the same thing needs to happen to many different pieces of input data. This is a nice thing to have happen, because it is generally easy to make happen in parallel. Data parallelism is something you should look for when you're trying to speed up your code. It is easy to parallelize, and many different types of problems can be reduced to them.

        This image represents a data-parallel process:
        ![data parallel job](https://upload.wikimedia.org/wikipedia/commons/a/a7/Sequential_vs._Data_Parallel_job_execution.png)
        The top describes a job happening in serial, while the bottom shows how one would divide up the top job into the four jobs represented on the bottom, but producing identical output.

        Now, let's get into task parallelism. What is task parallelism? Well, task parallelism is a form of parallelism where data is not distributed, but instructions are. Using our acronyms from before, it would be Multiple Instruction, Single Data (MISD). Task parallelism focuses on distributing tasks across different processors. In contrast to data parallelism which involves running the same task on different components of data, task parallelism is distinguished by running many different tasks at the same time on the same data.

        In a multiprocessor system, task parallelism is achieved when each processor executes a different thread (or process) on the same or different data. The threads may execute the same or different code. In the general case, different execution threads communicate with one another as they work, but is not a requirement. Communication usually takes place by passing data from one thread to the next as part of a workflow.

        As a simple example, if a system is running code on a 2-processor system (CPUs "a" & "b") in a parallel environment and we wish to do tasks "A" and "B", it is possible to tell CPU "a" to do task "A" and CPU "b" to do task "B" simultaneously, thereby reducing the run time of the execution.

        The image below represents a comparison between task and data parallelism, with task parallelism on the left and data parallelism is on the right.

        ![parallelism comparison](http://lh3.googleusercontent.com/8hC1WoryRSmlqL7iQ3VtQwRnCLeO2Wa5zejBw3pmUlXxLpoqhRC2OhNPLZTPfz5OlD4tdmt7V6YqXqBqYmSgg_CQakRHs6IGbwtaliLuUqj7pM6_gCcSbZKSnwoUXaVZnZ8s2pWz4nVxTDnovHjjTxCdpu7p6jly4-Z7ddBH5G43k7ffP8Y-3tXL-o-4UJ-7LJj62Oh7oHEOybzfuw1_MfbDN6N_utlGwuWh1rvy97Niu9Oy9EUS4E7t4N6aJkYUlZptNY7bxC8i0MixcXnCxkH0PtRv7_P-eubpTl2VyKFBEqPyOTsq_8rpiSLsD0vVNjMB7erjSihoNpTI24891t-TOgkJIb2ShZjzUVeyJQQqT2f7tkLCGYRfr8E2V60evnzFw5jyvr06tmImsAvtL2jvo7mduggeusz7HGu6KpUL1APfqt9hFqeJIZOYxUWc-lk7tJ_-XVvhiU4ouwDsKJB2ZhSt_leZODmmE76JtFBSnikFv4ip4YHlt3TwVcsjYF5QNTDHV_rX7CupI2ODOV9gV7MHE6bEkd06edUP7kclqojT1H4C1OUS_yHqHu-3aEpOkuzF-65KGLZ9gRIQL-hyLAGcKJw=w300-h225-no)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 6.3 - Basic Task and Data Parallelism
        In this example, we will design one workflow that exhibits task parallelism and one that exhibits data parallelism. A nice real world example of something that is not only a nice simple example of a parallel algorithm, but is also an example of a real world computational task that comes up a lot is matrix multiplication. This image displays how data parallelism comes up in the problem of matrix multiplication:
        ![matrix multiplication data parallelism](https://upload.wikimedia.org/wikipedia/commons/6/68/Data_Parallelism_in_matrix_multiplication.png)
        You can see that after some basic serial computation to set up the right hand side of the equation, each one of the cells in the 3x2 matrix on the right hand side can be computed concurrently or in parallel. This represents data parallelism because the same multiplication and addition instructions are used on different data. For multiplication, we can divide matrix A and B into blocks along rows and columns respectively. This allows us to calculate every element in matrix C individually thereby making the task parallel. 

        For example: $A(m \times n) \cdot B (n \times k)$ can be finished in $O(n)$ time instead of $O(m*n*k)$, when executed in parallel using $m*k$ processors. (This is the example from the image).

        As our example of task parallelism, we are going to use a very simple workflow - computing the square root of many numbers in parallel and computing the square of many numbers concurrently. This way, we have a single instruction (sqrt) which is run on some data, and a second instruction (square) which is run on other data. Based on the fact that all of the input data is in one data pool, a list, this is task parallelism because we are applying different tasks to data in different parts of the same data pool.
        """
    )
    return


@app.cell
def _(multiprocessing, random):
    from itertools import starmap, repeat
    from operator import mul

    def calc_row_of_product_matrix(a_row, b, izip=zip):
        """Calculate a row of the product matrix P = A * B
        Arguments:
          a_row is af A
          b is the B matrix
        returns the corresponding row of P matrix"""
        return map(lambda col: sum(starmap(mul, zip(a_row, col))), zip(*b))

    def eval_func_tuple(f_args):
        """Takes a tuple of a function and args, evaluates and returns result"""
        return f_args[0](*f_args[1:])

    class multimatrix(list):

        def __mul__(self, b, izip=zip, repeat=repeat):
            """Concurrent matrix multiplication with multiprocessing.Pool. """
            _pool = multiprocessing.Pool(multiprocessing.cpu_count())
            return _pool.map(eval_func_tuple, izip(repeat(calc_row_of_product_matrix), self, repeat(b)))

    class itermatrix(list):

        @staticmethod
        def sumprod(row, col, sum=sum, starmap=starmap, mul=mul):
            """Returns sumproduct of two vectors."""
            return sum(starmap(mul, zip(row, col)))

        def __mul__(self, b, imap=map, izip=zip):
            """Matrix multiplication returning iterable of iterables"""
            return map(lambda row: map(lambda col: itermatrix.sumprod(row, col), zip(*b)), self)

    def iterate_results(result):
        """Iterate over iterable of iterables,
        and returns elements in list of lists.
        Usage: if you want to run the whole calculation at once:
        p = iterate_results(itermatrix([[1, 3], [-5, 6], [2, 4]]) * itermatrix([[1, 4], [8, 7]]))"""
        return [[col for col in row] for row in result]

    def random_v(K=1000, min=-1000, max=1000):
        """Generates a random vector of dimension N;
        Returns a list of integers.
        The values are integers in the range [min,max]."""
        return [random.randint(min, max) for k in range(K)]

    def random_m(N=1000, K=1000):
        """Generates random matrix. Returns list of list of integers."""
        return [random_v(K) for _n in range(N)]
    return iterate_results, itermatrix


@app.cell
def _(iterate_results, itermatrix):
    # magic command not supported in marimo; please file an issue to add support
    # %%time

    if __name__ == '__main__':
        a = [[1, 3], [-5, 6], [2, 4]]
        b = [[1, 4], [8, 7]]
        adotb = [[25, 25], [43, 22], [34, 36]]
        #A = multimatrix(a)
        #B = multimatrix(b)
        #prod = A * B
        #assert(adotb == prod)
        #print(prod, "multi test OK")
        A = itermatrix(a)
        B = itermatrix(b)
        iterprod = A * B
        listprod = iterate_results(iterprod)
        assert(adotb == listprod)
        print(listprod, "iter test OK")
    return


@app.cell
def _(multiprocessing):
    def sqrt(n):
        return _n ** 0.5

    def square(n):
        return _n ** 2
    run_serial = False
    if run_serial:
        sqrts = []
        squares = []
        for _i in range(1000000):
            if _i % 2 == 0:
                sqrts.append(sqrt(_i))
            else:
                squares.append(square(_i))
        print(sqrts[0:9])
        print(squares[0:9])
    _pool = multiprocessing.Pool(32)
    sqrts = _pool.map(sqrt, [_i for _i in range(1000000) if _i % 2 == 0])
    squares = _pool.map(square, [_i for _i in range(1000000) if _i % 2 == 1])
    print(sqrts[0:9])
    print(squares[0:9])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## IV. Maximum Speedups
        By now, you should understand that different types of algorithms can be helped by parallelization different amounts. Let's go ahead and quantify that. First, let's do some mathematical analysis and figure out how well we could do. The first major breakthrough in terms of parallel algorithm analysis is called [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law). Formulated by Gene Amdahl in 1967, Amdahl's Law allows us to compute theoretical maximum speedups based on how parallelizable our tasks are. In order to figure out your theoretical maximum, you need to know how many processors you have access to and what percentage of your program can be parallelized. For example, if a program needs 20 hours using a single processor core, and a particular part of the program which takes one hour to execute cannot be parallelized, while the remaining 19 hours (p = 0.95, where p is the fraction of the code that can be parallelized) of execution time can be parallelized, then regardless of how many processors are devoted to a parallelized execution of this program, the minimum execution time cannot be less than that critical one hour. Hence, the theoretical speedup is limited to at most 20 times (1/(1 − p) = 20). For this reason, parallel computing with many processors is useful only for highly parallelizable programs. This image sums up Amdahl's law for various values of p.
        ![amdahl's law](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/AmdahlsLaw.svg/640px-AmdahlsLaw.svg.png)

        Another parallel algorithm analysis law, the [Gustafson–Barsis Law](https://en.wikipedia.org/wiki/Gustafson%27s_law), gives the theoretical speedup in latency of the execution of a task at fixed execution time that can be expected of a system whose resources are improved. This law considers not how much faster your code can get, but instead how much more you can do with the same code in the same amount of time as you scale up your computer. Amdahl's law presupposes that the computing requirements will stay the same, given increased processing power. In other words, an analysis of the same data will take less time given more computing power.

        Gustafson, on the other hand, argues that more computing power will cause the data to be more carefully and fully analyzed: pixel by pixel or unit by unit, rather than on a larger scale. Where it would not have been possible or practical to simulate the impact of nuclear detonation on every building, car, and their contents (including furniture, structure strength, etc.) because such a calculation would have taken more time than was available to provide an answer, the increase in computing power will prompt researchers to add more data to more fully simulate more variables, giving a more accurate result.


        Amdahl's Law reveals a limitation in, for example, the ability of multiple cores to reduce the time it takes for a computer to boot to its operating system and be ready for use. Assuming the boot process was mostly parallel, quadrupling computing power on a system that took one minute to load might reduce the boot time to just over fifteen seconds. But greater and greater parallelization would eventually fail to make bootup go any faster, if any part of the boot process were inherently sequential.

        Gustafson's law argues that a fourfold increase in computing power would instead lead to a similar increase in expectations of what the system will be capable of. If the one-minute load time is acceptable to most users, then that is a starting point from which to increase the features and functions of the system. The time taken to boot to the operating system will be the same, i.e. one minute, but the new system would include more graphical or user-friendly features.

        This image sums up Gustafson's law for various values of p (same p as earlier):
        ![Theoretical maximum speedup](https://cdn.comsol.com/wordpress/2014/03/Graph-depicting-how-the-size-of-the-job-increases-with-the-number-of-available-processes.png)

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 6.4 - Timing Fibonacci and Monte Carlo
        Given all of that theory we just learned, what do you think the value of _p_ is for the Monte Carlo pi simulation? For the Fibonacci sequence generator? Think about it for a moment. Read ahead when you've got a guess. For the Monte Carlo Pi, the value of _p_ is very close to 1, while for the Fibonacci sequence generator, the value of _p_ is very close to 0. Hopefully the description and examples of those tasks make those values believable to you. In case they don't, we're going to time them right now, in this example. 
        """
    )
    return


@app.cell
def _(random):
    def monte_carlo_pi_part_1(n):
        _count = 0
        for _i in range(_n):
            x = random.random()
            y = random.random()
            if x * x + y * y <= 1:
                _count = _count + 1
        return _count
    return (monte_carlo_pi_part_1,)


@app.cell
def _(Pool, monte_carlo_pi_part_1, multiprocessing):
    _n = 10000000
    _np = multiprocessing.cpu_count()
    _part_count = [_n // _np for _i in range(_np)]
    _pool = Pool(processes=_np)
    _count = _pool.map(monte_carlo_pi_part_1, _part_count)
    print('Esitmated value of Pi: {} '.format(sum(_count) / (_n * 1.0) * 4))
    return


@app.cell
def _(multiprocessing, time):
    def fib_1(n):
        if _n == 0 or _n == 1:
            return 1
        return fib(_n - 1) + fib(_n - 2)

    def calc_single_fib(i):
        before = time.time()
        res = fib_1(_i)
        after = time.time()
        _tot = after - before
    _pool = multiprocessing.Pool(32)
    tasks = _pool.map(calc_single_fib, range(0, 40, 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that even though we run each distinct fibonacci number in parallel, we can't parallelize the underlying function at all, so it's still very slow.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercise 6. Pi Over Many Nodes or Fibonacci Over Many Nodes
        Using Parsl, implement a fibonacci sequence generator and a mote carlo pi simulation. Run it on the entire cluster. Which one runs faster? Which one seems to scale better?
        """
    )
    return


@app.cell
def _():
    # Your code goes here
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

