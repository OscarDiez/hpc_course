import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Ch. 3. Parallel Algorithms
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## I. Trivially Parallelizable Algorithms
        In order to properly make use of an HPC system or other parallel computing environment, you need to be able to make your code able to run in parallel. As we saw with our monte carlo pi example, even making the same algorithm run in parallel can make huge changes in terms of performance. If a job running on an HPC system has access to hundreds of cores, but is only written to run in serial, it will not make use of the computational resources it has available to it. In theory, if your task is completely parallelizable, then if you scale it up by a factor of _n_ cores, it will speed up _n_ times. That is to say, if you run a completely parallelizable job on 20 cores, it will run up to 20 times faster than on one core. 
        A graph of this is below: 

        ![Theoretical maximum speedup](https://cdn.comsol.com/wordpress/2014/03/Graph-depicting-how-the-size-of-the-job-increases-with-the-number-of-available-processes.png)

        This graphic represents the theoretical maximum speedup of various types of HPC jobs, where the x-axis represents the number of cores the job is run on, the y axis represents how much faster the job runs (i.e. if the y-value is 20, the job is 20 times faster than single core), and the value of _phi_ represents the fraction of the job that can be parallelized. The mathematical principle this is based on is called [The Gustafson-Barsis Law](https://en.wikipedia.org/wiki/Gustafson%27s_law)

        The class of algorithms for which the value of _phi_ is 1, that is to say the class of algorithms that scale perfectly, is known as the "trivially parallelizable" class. This means that you can expect linear performance scaling as you increase number of cores linearly. These algorithms are, rather unsurprisingly, very easy to make parallel. Because of this, people using HPC systems often try to reduce their workloads to different "building blocks" made up of trivially parallelizable algorithms. A Trivially parallelizable algorithm is defined by task-independence. That is to say, if you break up an algorithm into tasks, in order for that algorithm to be trivially parallelizable, each task must not depend on the output of any other task. The image below represents a trivially parallelizable algorithm:

        ![trivially parallel](http://matthewrocklin.com/slides/images/embarrassing-process.png)

        Each set of input data goes into a process, and comes out changed. None of them depend on what is happening in other simultaneous processes.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1 - Generation of Data
        As an example of a trivially parallelizable algorithm, we are going to generate lots of numbers in serial and in parallel. This will also teach you a very practical skill with python's `multiprocessing` library: how to "unroll" a trivially parallelizable loop into processes that can run simultaneously.
        """
    )
    return


@app.cell
def _():
    with open('./data/datagen.out', 'w') as _file:
        for _i in range(1, 100001):
            _file.write(str(_i) + '\n')
    with open('./data/datagen.out') as _file:
        print(len(_file.readlines()))
    return


@app.cell
def _():
    from multiprocessing import Pool
    import time
    _file = open('./data/datagenparallel.out', 'w')

    def process_single(i):
        _file.write(str(_i) + '\n')
    _pool = Pool(32)
    _tasks = [_pool.apply_async(process_single, (j,)) for j in range(1, 1000)]
    time.sleep(3)
    _file.close()
    with open('./data/datagenparallel.out') as file2:
        print(len(file2.readlines()))
    return Pool, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can see that writing the code in parallel is not much more complex than writing it in serial. In this case, the task is carried out so quickly in serial that it is not really worth it to parallelize, but there are a multitude of real-life scenarios where a workflow, or at least part of a workflow is trivially parallelizable and it makes sense to do so.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## II. Monte Carlo Simulations
        Remember in the last section how I made a fuss about how, even though it seems really unlikely that real world applications would ever be even close to trivially parallelizable, it's a goal that many scientists and other HPC users aim for? Well, one of the tools these people commonly use to move that direction is the _monte carlo simulation_. We've already used monte carlo simulations to compute a numerical value of pi experimentally, but we haven't really strictly defined just what exactly a monte carlo algorithm _is_, so let's do that now. A _monte carlo algorithm_ is an algorithm that attempts to produce an answer to a question by simulating many possible outcomes. For example, our monte carlo pi simulates throwing a dart at a dartboard over and over, and then uses the underlying geometry of the situation to extract a value from those darts' randomly selected locations. A monte carlo algorithm is often trivially parallelizable, because random numbers, by definintion, can always be generated independently, that is with no input from each other. For this reason, many HPC users attempt to use the monte carlo method when attempting to solve problems with "real" values (i.e. numerically).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2 - Monte Carlo Frog Hop Simulation
        To illustrate the scalability of monte carlo algorithms, we're going to use an example problem which my professor Jeff Ely uses to test the mettle of CS 1 students. Imagine a frog who is sitting on an infinite flat plane. The frog always makes jumps of length 1 unit, in a randomly selected direction. What is the probablility that the frog will be within 1 unit of its original location after _n_ jumps?

        You may realize that we can solve this problem with no programming at all, just math. In this case, this is true, and we can use this fact to help us check our work. Often, especially in the fields of physics and differential equations, there are problems which cannot be solved other than numerically. We will solve this problem first in serial and then we will parallelize our solution later.
        """
    )
    return


@app.cell
def _():
    import math
    import random

    def monteCarlo(numJumps):
        numTries = 100000
        lenJump = 1
        numSuccesses = 0
        for j in range(0, numTries, 1):
            frogPosition = [0.0, 0.0]
            for _i in range(0, numJumps, 1):
                theta = random.uniform(0, 2 * math.pi)
                frogPosition[0] += lenJump * math.cos(theta)
                frogPosition[1] += lenJump * math.sin(theta)
            frogMagnitude = (frogPosition[0] ** 2 + frogPosition[1] ** 2) ** 0.5
            if frogMagnitude <= 1.0:
                numSuccesses += float(1)
        successRate = float(numSuccesses / numTries)
        print('If the frog jumps %s times, it will land in the original circle approximately %s times, representing a success rate of %s' % (numJumps, int(numSuccesses), successRate))
    for _i in range(20):
        monteCarlo(_i)
    return math, random


@app.cell
def _(Pool, math, random):
    def outer_loop(numJumps):
        numTries = 100000
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
        print('If the frog jumps %s times, it will land in the original circle approximately %s times, representing a success rate of %s' % (numJumps, int(numSuccesses), successRate))
    _pool = Pool(32)
    _tasks = [_pool.apply_async(outer_loop, (j,)) for j in range(1, 20)]
    _tasks[-1].get()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As you can see, the parallel version is faster than the serial version
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## III. Single Instruction, Multiple Data
        A wide variety of different HPC workflows include some slight variation on what's called a _Single Instruction, Multiple Data_ algorithm. _Single Instruction, Multiple Data_, or SIMD algorithms are considered "nice" algorithms in that it is very easy to write scalable code for them. This code often behaves as well as trivially parallelizable algorithms, or with specialized hardware accelerators (like General Purpose GPUs), even better. HPC accelerators and heterogeneous computing is an extremely rich and wildly fascinating topic.
        The image below represents a SIMD process:


        ![simd-architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/SIMD.svg/500px-SIMD.svg.png)

        In this example, the "instruction pool" provides each of the processing units, labeled "PU", the same instruction, and arbitrary data from the "data" pool gets fed into any of the processing units as they become available

        SIMD algorithms are certainly something you should look for when you're trying to speed up your code. They are easy to parallelize, and many different types of problems can be reduced to them. In addition, they can often be parallelized extremely effectively, because most modern processors have hardware specifically designed to perform SIMD instructions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 3 - Vector Addition
        A common use case of parallelism in HPC is for performing vector math. Vector arithmetic comes up in all sorts of science, engineering, and other HPC applications. Many different pieces of science and math can be reduced, with some clever approximation, to vector math. In this example, we will take the sums of large vectors in serial and in parallel. Vector addition is an easily parallelizable task, because the way vector sums are computed is by taking pairwise sums of each corresponding vector subscript.
        """
    )
    return


@app.cell
def _():
    _vec_a = [3] * 22000000
    _vec_b = [4] * 22000000
    _vec_c = [0] * 22000000
    for _i in range(len(_vec_a)):
        _vec_c[_i] = _vec_a[_i] + _vec_b[_i]
    print(len(_vec_c))
    return


@app.cell
def _():
    _vec_a = [3] * 22000000
    _vec_b = [4] * 22000000
    _vec_c = [0] * 22000000

    def add_single(i):
        _vec_c[_i] = _vec_a[_i] + _vec_b[_i]
    import multiprocessing
    _pool = multiprocessing.Pool(32)
    _pool.map(add_single, range(len(_vec_a)))
    print(len(_vec_c))
    return (multiprocessing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## IV. Multiple Instruction, Multiple Data
        SIMD algorithms are extremely performant, scalable, and easy to work with, but sometimes, you have a data pipeline where different things need to happen depending on what the data looks like. This scenario is often referred to as _Multiple Instruction, Multiple Data_, or MIMD. MIMD algorithms are much more flexible than SIMD ones, for obvious reasons. Every modern multicore computer can perform as a MIMD computer, to varying efficiencies. As of 2018, well over 95% of the TOP500 supercomputers use MIMD architectures as the basis for their computation. Of course, processors designed for MIMD computation can do SIMD computation as well, but they also have the ability to apply different kinds of logic based on data values. This ability makes MIMD extremely flexible for many diverse types of workloads. That detour aside, The image below represents a SIMD process:


        ![mimd-architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/MIMD.svg/500px-SIMD.svg.png)


        In this image, multiple different types of data can come out of the "data pool" and are provided to the "processing units" along with various types of instructions from the "instruction pool." Those instructions are not necessarily the same as each other, and can depend on the value of the data, randomization, or any other arbitrary logic.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 4 - Asynchronous Branching Logic
        In this example, we are going to first, generate random numbers in parallel, and then, put them into separate files based on whether they are even or odd, and then finally, sort the files and print them out. We are going to use `multiprocessing` to perform each step in parallel.
        """
    )
    return


@app.cell
def _(multiprocessing, random):
    import os
    _path = './data'
    if os.path.isfile(_path + '/evenodd.out'):
        os.remove(_path + '/evenodd.out')
    if os.path.isfile(_path + '/even.out'):
        os.remove(_path + '/even.out')
    if os.path.isfile(_path + '/odd.out'):
        os.remove(_path + '/odd.out')

    def gen_num(i):
        _file = open(_path + '/evenodd.out', 'a')
        _file.write(str(random.randint(0, 100)) + '\n')
        _file.close()
    _pool = multiprocessing.Pool(32)
    _pool.map(gen_num, range(100))

    def bin_out(item):
        if int(item) % 2 == 0:
            file2 = open(_path + '/even.out', 'a')
            file2.write(str(item) + '\n')
            file2.close()
        else:
            file3 = open(_path + '/odd.out', 'a')
            file3.write(str(item) + '\n')
            file3.close()
    data = open(_path + '/evenodd.out', 'r').readlines()
    _pool = multiprocessing.Pool(32)
    _pool.map(bin_out, data)

    def keyfunc(line):
        try:
            return int(line)
        except:
            return 0
    for _file in [_path + '/even.out', _path + '/odd.out']:
        with open(_file) as fin:
            content = sorted(fin, key=keyfunc)
        with open(_file, 'w') as fout:
            fout.writelines(content)
    for _file in [_path + '/even.out', _path + '/odd.out']:
        with open(_file) as fin:
            print([x.strip() for x in fin.readlines() if x != '\n'])
            print('\n')
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## V. Concurrency vs. Parallelism
        You may have heard the terms _concurrency_ and _parallelism_ before. They are often confused, and this makes sense because they both relate to running more than one process at a time. Concurrency is the composition of independently executing processes, while parallelism is the simultaneous execution of (possibly related) computations. Concurrency is about dealing with lots of things at once. Parallelism is about doing lots of things at once. Another reason they often get conflated is that they are often used together. 

        A purely SIMD algorithm is a parallel environment, while any kind of MIMD algorithm includes at least some concurrency. Another time when concurrency is useful is when the workload you are trying to run involves interprocess communication, waiting for hard disk access, or waiting for network access. The reason for this is that it takes a lot of time to access the network, and a clever programmer will allow another process to use the spare CPU cycles that would be otherwise spent waiting. Hopefully, you can see how there is potential for overlap of parallelism and concurrency. 

        Imagine a scenario where you want to generate files by connecting to an external service (maybe you're asking the Google maps API where something is), and then you want to perform some computation on those results. A purely parallel approach would be to make all of the Google API requests (all in parallel), wait for them all to finish, and then do all of the computation on each response, again in parallel. A better approach, one that uses concurrency, would be to make all of the API requests, and while some of the slower requests are waiting for their responses, allow the faster requests to begin their computation.

        The following image describes parallelism and concurrency:

        ![parallelism and concurrency](https://pbs.twimg.com/media/DSFCqf2U8AAjgqI.jpg)

        Using this image as a reference, we can imagine a scenario where there are multiple CPUs, as in the bottom image, and there are multiple queues per CPU, as in the top image. This would be a combination of parallelism and concurrency.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 5a - Network Interaction and IO-Bound Tasks
        As mentioned in the previous section, many of the tasks that lend themselves to concurrent execution are tasks with network or filesystem IO. This is because the amount of time it takes to connect to the external internet, or even the time it takes to connect to the local filesystem is much longer than the CPU's instruction cycle. This causes long periods of time (up to many seconds or minutes, depending on the task), where the CPU is just waiting for things to happen. Because of this, you are able to run many of these tasks all at the same time, because some can process while others wait, and vice-versa. In situations like this, you can run many more of these tasks than you have cores on your computer and expect reasonable performance speedups.

        In this example, we will create one sleep job and demonstrate that we can run sleep jobs concurrently.
        """
    )
    return


@app.cell
def _(multiprocessing, os, time):
    import threading
 
    NUM_WORKERS = 32
 
    def only_sleep():
        """ Do nothing, wait for a timer to expire """
        print("PID: %s, Process Name: %s, Thread Name: %s" % (
            os.getpid(),
            multiprocessing.current_process().name,
            threading.current_thread().name)
        )
        time.sleep(1)
    return NUM_WORKERS, only_sleep, threading


@app.cell
def _(NUM_WORKERS, only_sleep, threading, time):
    serial_run = False
    if serial_run:
        _start_time = time.time()
        for _ in range(NUM_WORKERS):
            only_sleep()
        _end_time = time.time()
        print('Serial time= {}'.format(_end_time - _start_time))
    _start_time = time.time()
    _threads = [threading.Thread(target=only_sleep) for _ in range(NUM_WORKERS)]
    [thread.start() for thread in _threads]
    [thread.join() for thread in _threads]
    _end_time = time.time()
    print('Threads time= {} seconds'.format(_end_time - _start_time))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 5b - Number Crunching and CPU-Bound Tasks
        Algorithms which can be run concurrently to great effect are often nice to work with. This is because they are, by definition, easy to speed up without too many resources. I like to refer to another class of easy to speed up tasks as _number crunching_. Tasks that fit into this category must a) have little or no input data, b) produce easy to handle amounts of output data, and c) use up at least an entire compute core for most of the time it needs to be running. Because of these requirements, Number crunching tasks are easy to speed up through parallelization, by adding more cores to a pool. They are slightly harder to speed up than tasks from the previous example, because they require additional hardware for more speedup, but they are still easy to speed up overall.

        In this example, we will create a number crunching task and show that it benefits from being run in parallel.
        """
    )
    return


@app.cell
def _(multiprocessing, os, threading):
    # Define number crunching workload


    def crunch_numbers():
        """ Do some computations """
        print("PID: %s, Process Name: %s, Thread Name: %s" % (
            os.getpid(),
            multiprocessing.current_process().name,
            threading.current_thread().name)
        )
        x = 0
        while x < 10000000:
            x += 1
    return (crunch_numbers,)


@app.cell
def _(NUM_WORKERS, crunch_numbers, multiprocessing, threading, time):
    thread_run = False
    if thread_run:
        _start_time = time.time()
        _threads = [threading.Thread(target=crunch_numbers) for _ in range(NUM_WORKERS)]
        [thread.start() for thread in _threads]
        [thread.join() for thread in _threads]
        _end_time = time.time()
        print('Threads time= {} seconds'.format(_end_time - _start_time))
    _start_time = time.time()
    processes = [multiprocessing.Process(target=crunch_numbers) for _ in range(NUM_WORKERS)]
    [process.start() for process in processes]
    [process.join() for process in processes]
    _end_time = time.time()
    print('Parallel time = {} seconds'.format(_end_time - _start_time))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## VI. Dataflows

        A _dataflow_ is a set of different algorithms, which may or may not be parallel, that all fit together in some way, with the output of some parts leading to the input of other parts. Most real world HPC applications are not just simple SIMD or MIMD blocks, but are instead a number of different pieces of code, all of which produce data based on other applications or input data. Each of those individual pieces of code may be SIMD or concurrent, or trivially parallel, or monte carlo, or any other kind of code, but the important part is understanding how they all work together. The dataflow is an incredibly powerful way of "gluing" many programs together into one program that carries out exactly what you want it to. The image below is a representation of a dataflow:

        ![basic dataflow](http://www.digitaleng.news/de/wp-content/uploads/2016/10/HPC-Workflow.jpg)


        In this case, data is first generated, then the quality of that data is assured, and then the data gets computed on, creating some output, and finally, that data gets visualized and consolidated to a human-readable format. This is a fairly common type of dataflow. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 6 - `sleep_fail` dataflow
        The dataflow presented in the previous section is a relatively typical dataflow, but it's also farily complex. To give you a more gentle introduction to working with dataflows, we are going to design a dataflow called `sleep_fail`. This dataflow consists of a data generation layer, in which we define a function which sleeps for _n_ (user input) seconds before failing with a probability _p_ (also user input). Then, it writes whether it failed or succeeded to a file. Then, we have a data consolidation layer, which reads that file and prints out how many times it ran, succeeded, and failed. The resulting workflow looks a bit like this:
        ```
            sleep1  sleep2 ...sleepN
              |       |        |
              V       V        v
               \      |       /
                \     |      /
                  sleep_Final
        ```
        Note that this dataflow is serial. This is intended so that we can introduce you to the concept of a dataflow without worrying about parallelism.
        """
    )
    return


@app.cell
def _(random, time):
    def sleep_fail(n, p, path):
        time.sleep(n)
        if random.random() < p:
            _file = open(_path, 'a')
            _file.write('Exception\n')
            _file.close()
            raise Exception
        else:
            _file = open(_path, 'a')
            _file.write('Success\n')
            _file.close()

    def summarize_sleeps(path):
        _file = open(_path, 'r')
        succeed = 0
        exceptio = 0
        for line in _file.readlines():
            if 'Exception' in line:
                exceptio += 1
            elif 'Success' in line:
                succeed += 1
        print('Failed {} Times'.format(exceptio))
        print('Passed {} Times'.format(succeed))
        print('Ran {} Times'.format(succeed + exceptio))
    return sleep_fail, summarize_sleeps


@app.cell
def _(os, sleep_fail, summarize_sleeps):
    _path = './data/sleep_fail.out'
    if os.path.isfile(_path):
        os.remove(_path)
    for _i in range(500):
        try:
            sleep_fail(0, 0.2, _path)
        except Exception as e:
            pass
    summarize_sleeps(_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Optional Exercise: Write a dataflow that estimates the value of _e_ 
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

