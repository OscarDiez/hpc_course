import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # M1.P1 Introduction
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## I. What is High-Performance Computing?

        Most people believe that High-Performance Computing (HPC) is defined by the architecture of the computer it runs on. In standard computing, one physical computer with one or more cores is used to carry out a task. All of these cores can access a pool of [Shared Memory](https://en.wikipedia.org/wiki/Shared_memory). Shared memory makes it trivial for two or more programs or parts of a single program to understand what other programs are doing. The following diagram represents a multicore single computer working in this fashion: ![multicore architecture diagram](https://i.pinimg.com/originals/22/31/f8/2231f856a5341e19526d089e1ffbe630.jpg "Logo Title Text 1")


        On the other hand, most high performance systems are clusters which consist of many separate multicore computers which are all connected over a network. In this configuration, the individual parts of a program running on separate physical machines are not easily able to communicate without connecting over the network. The following diagram represents a cluster of multicore computers working as a single cluster:  ![HPC cluster architecture diagram]

        (https://supercomputingwales.github.io/SCW-tutorial/fig/cluster-generic.png "Logo Title Text 1") 

        Because of the dependence on networking, programming for HPC environments requires special considerations that normal programming does not.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1.1 - Computing Pi on a Standard Computer
        As an example of "traditional" computation, we will now compute the numerical value of Pi through the monte carlo simulation method.

        Monte carlo simulations are a class of computational algorithms that rely on repeated random sampling to obtain numerical results. We will discuss them in more detail in the parallel algorithms notebook. A very common monte carlo simulation uses some basic geometry to estimate the numerical value of pi. You can find more information on Monte Carlo simulations here: https://www.youtube.com/watch?v=7ESK5SaP-bc

        For this, we imagine a circle of radius 1 inscribed in a square. Then, we choose random points in the square, and classify them by whether they are inside the circle or outside. This gives us an estimate of the areas of the square and the circle. Then, we take the ratio of one to the other and we have an estimation of pi. If this doesn't make any sense to you, don't worry. This image may help you understand a bit better: ![Monte Carlo Pi]

        (https://ds055uzetaobb.cloudfront.net/image_optimizer/aabd5727316301f18f53bd4cbc63914fed0bcb2c.gif "Logo Title Text 1")


        In the example below, we calculate pi by monte carlo simulation.
        Try to change the value of variable total from 100 to 10000000 .

        In Jupyter, you can change the code and run it using the play button in the top of the page. 
        """
    )
    return


@app.cell
def _():
    import random as r
    import math as m
    _inside = 0
    _total = 10000
    for _i in range(0, _total):
        _x2 = r.random() ** 2
        _y2 = r.random() ** 2
        if m.sqrt(_x2 + _y2) < 1.0:
            _inside = _inside + 1
    _pi = float(_inside) / _total * 4
    print('Estimated value of Pi: {} '.format(_pi))
    return m, r


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## II. Why is HPC important?
        High performance computing opens the door to large scale data analysis, computational science, and research computing. It is useful in a number of scenarios, including where software is too time-critical, too performance critical, or simply too big to run on a traditional system.

        ![HPC Applications](https://ec.europa.eu/information_society/newsroom/image/document/2021-5/hpc_applications_3D20F502-F32E-357F-31E23744FC4EE2C3_73074.jpg "")




        Let's take a look at a few examples of scenarios where you would need an HPC System or an HPC System drastically changes your process.


        - **Scenario 1: Predicting Natural Disasters and Understanding Climate Change :** A key field where HPC has delivered a transformational impact is Earth sciences. Supercomputing is frequently used to study climate change and its impact. Research organizations around the world rely on HPC to predict weather phenomena and enable highly accurate hyperlocalized forecasts. A crucial broader application area of these foundational domains is emergency preparedness, where HPC models are used to predict aspects of natural disasters such as intensity and impact of earthquakes, path and ferocity of hurricanes, direction and impact of tsunamis, and more. The climate is ever changing, with increasing threats of intense hurricanes, heatwaves, and other extreme events necessitating the need for higher-fidelity computational models and more supercomputing capabilities


        ![Weather Models](https://smd-prod.s3.amazonaws.com/science-red/s3fs-public/styles/large/public/mnt/medialibrary/2015/08/03/WeatherFocusGPM.png?itok=0duoMhY0 "")


        - **Scenario 2: Designing a New Car or Plane:** You're a brand new aerospace engineer working for the Mercedes-Benz Formula One team. You have the off season (usually between December and May, or about five months) to design a new car which is better than all the cars that beat you last year. Traditionally, the way to do this is start with a small model, put it in a wind tunnel, evaluate it, and repeat this process. Then, you slowly scale up to bigger models and eventually start building concept cars. However, you only have five months, and each model may take a month to design and produce. You simply don't have time. Instead, you get started with your HPC system and start creating some [Computational Fluid Dynamics](https://en.wikipedia.org/wiki/Computational_fluid_dynamics) models which you can then use to create your new car with plenty of time to spare. The image below is the output of a CFD model. ![CFD model of car]


        (https://upload.wikimedia.org/wikipedia/commons/f/fa/Verus_Engineering_Porsche_987.2_Ventus_2_Package.png " ")


        - **Scenario 3: Personalized Medicine and Drug Discovery:** Life sciences are another major vertical segment that relies on HPC technologies in various application areas. Supercomputing is used by researchers and enterprises for genome sequencing and drug discovery. Pharmaceutical companies often deploy supercomputers to accelerate the process of drug discovery using various molecular dynamic simulation methodologies. Using HPC and molecular dynamics simulations researchers are able to design new drugs and virtually test effectiveness, enabling significant optimization of the research process while resulting in safer and more effective drugs. HPC is also used to develop virtual models of human physiology (e.g., heart, brain, etc.), which enable scientists and researchers to understand ailments and potential treatments better. Increasingly life sciences researchers and companies are engineering new methodologies combining genome sequencing and drug discovery to enable new and more effective forms of personalized medicine that could cure some of the most challenging diseases.


        ![computational climate research](https://www.cbkscicon.com/wp-content/uploads/2019/09/small_crop_Screen-Shot-2018-03-08-at-17.17.33-1-300x300.png)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1.2 - Multiplying Matrices in Parallel

        Though it's not as large-scale as any of the examples mentioned above, a very applicable and useful application of parallel programming is matrix multiplication. Runtime of multiplying two _n_ by _n_ matrices is in the complexity class O(_n<sup>3</sup>_). This means that a good parallel algorithm which can make use of multiple cores for this process is very important. An example of a parallel algorithm to multiply matrices is below.
        """
    )
    return


@app.cell
def _():
    def matmult(a,b):
        zip_b = zip(*b)
        # uncomment next line if python 3 : 
        # zip_b = list(zip_b)
        return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
                 for col_b in zip_b] for row_a in a]

    x = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    y = [[1,2],[1,2],[3,4]]

    import numpy as np # I want to check my solution with numpy

    mx = np.matrix(x)
    my = np.matrix(y)
    return matmult, mx, my, x, y


@app.cell
def _(matmult, mx, my, x, y):
    print(matmult(x,y))
    print(mx * my)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## III. Multiprocessing
        One of the reasons this course is written in python is that python is an easy to use language which is commonly used for computational science and other HPC applications. Another reason that Python was chosen for this course is Python's powerful _multiprocessing_ library. _Multiprocessing_ allows users to open subprocesses within python in order to run many different snippets of python code all at once. In the following examples, we will use _multiprocessing_ to speed up our computations. It is true that Python has a GIL (global interpreter lock) which keeps it from natively running truly parallel programs, but with multiprocessing, we can get around this at the expense of heavier RAM overhead. The purpose of this course is not necessarily to write the fastest possible code, but more to demonstrate the techniques, methods, and tools for doing that in the future, and Python is an easily understandable and highly readable language. The following image is a description of how multiprocessing can help speed up Python programs: ![Multiprocessing Diagram](https://sebastianraschka.com/images/blog/2014/multiprocessing_intro/multiprocessing_scheme.png "")
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1.3 - Basic Parallelization with Multiprocessing
        In this example, we are going to create random strings in parallel and in serial. This is a good task to parallelize because each random string does not depend at all on the random strings that came before it. We will discuss in greater detail what makes tasks better candidates for parallelization in topic "algorithm analysis".
        """
    )
    return


@app.cell
def _():
    import random
    import string
    import multiprocessing
    random.seed(123)
    output = multiprocessing.Queue()

    def rand_string(length, output):
        """ Generates a random string of numbers, lower and uppercase chars. """
        rand_str = ''.join((random.choice(_string.ascii_lowercase + _string.ascii_uppercase + _string.digits) for _i in range(length)))
        output.put(rand_str)
    return multiprocessing, output, rand_string


@app.cell
def _(output, rand_string):
    import time
    _NUM_STRINGS = 10000
    processes = []
    _before = _time.time()
    for _ in range(_NUM_STRINGS):
        processes.append(rand_string(5, output))
    _results = [output.get() for p in processes]
    _after = _time.time()
    print('Generated {} strings in {} seconds'.format(_NUM_STRINGS, _after - _before))
    return


@app.cell
def _(random_1):
    import random
    import string
    import time
    from multiprocessing import Pool

    def rand_string_1(length):
        """
        Generate a random string.
    
        :param length: int, The length of the random string to be generated.
        :return: str, Generated random string.
        """
        rand_str = ''.join((random_1.choice(_string.ascii_lowercase + _string.ascii_uppercase + _string.digits) for _ in range(length)))
        return rand_str
    _before = _time.time()
    _NUM_STRINGS = 10000
    with Pool() as _pool:
        _results = _pool.map(rand_string_1, [5] * _NUM_STRINGS)
    _after = _time.time()
    print('Generated {} strings in {} seconds'.format(_NUM_STRINGS, _after - _before))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As you can see, the task runs more than five times faster in parallel than it does in serial
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## IV. Parallelism and HPC
        Parallelism is at the core of any HPC system. The way that HPC systems can be many hundreds of thousands of times faster than traditional systems is through massive parallelism. Some HPC systems have a total of many millions of cores, distributed among many systems (known as 'nodes'), as compared to the 4-8 of a standard modern desktop. This massive hardware parallelism, combined with some clever parallel algorithms, can lead to amazing processing power. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1.4 - Monte Carlo Pi in Parallel
        In this example, we will compute pi through monte carlo approximation, similar to the way we did it before, except this time we will do it in parallel. As mentioned earlier, the monte carlo pi calculation is a very easily parallelized algorithm, and there should be a large speedup when it is run in parallel.
        """
    )
    return


@app.cell
def _(random_1):
    from multiprocessing import Pool

    def monte_carlo_pi_part(n):
        _count = 0
        for _i in range(_n):
            x = random_1.random()
            y = random_1.random()
            if x * x + y * y <= 1:
                _count = _count + 1
        return _count
    return (monte_carlo_pi_part,)


@app.cell
def _(Pool_1, monte_carlo_pi_part, multiprocessing):
    np_1 = multiprocessing.cpu_count()
    print('You have {0:1d} CPUs'.format(np_1))
    _n = 10000000
    _part_count = [_n // np_1 for _i in range(np_1)]
    _pool = Pool_1(processes=np_1)
    _count = _pool.map(monte_carlo_pi_part, _part_count)
    print('Esitmated value of Pi: {} '.format(sum(_count) / (_n * 1.0) * 4))
    return (np_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## V. Power and Speed Comparison
        One definition of HPC has to do with scale. Some people believe that an HPC system is defined by how powerful it is rather than by how it is designed. Computational power has increased an incredible amount recently and HPC scale systems are now easily accessible to many people. The following graphic shows how powerful modern systems can be: ![hpc system power comparison](https://i.imgur.com/frXsxpz.png)

        As you might guess, parallel algorithms have the potential to be much faster than serial algorithms. In order to show this, we are going to use the two versions of monte carlo pi that we have designed in this notebook and time them against each other. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1.5 - Pi in Serial vs Parallel
        In this example, we will pit against each other our parallel and serial pi approximation calculations. To do this, we will use the jupyter "magic" `%%time` to time the two algorithms.
        """
    )
    return


@app.cell
def _(random_1):
    def monte_carlo_pi_part_1(n):
        _count = 0
        for _i in range(_n):
            x = random_1.random()
            y = random_1.random()
            if x * x + y * y <= 1:
                _count = _count + 1
        return _count
    return (monte_carlo_pi_part_1,)


@app.cell
def _(Pool_1, monte_carlo_pi_part_1, np_1):
    _n = 10000000
    _part_count = [_n // np_1 for _i in range(np_1)]
    _pool = Pool_1(processes=np_1)
    _count = _pool.map(monte_carlo_pi_part_1, _part_count)
    print('Esitmated value of Pi: {} '.format(sum(_count) / (_n * 1.0) * 4))
    return


@app.cell
def _(m, r):
    _inside = 0
    _total = 10000000
    for _i in range(0, _total):
        _x2 = r.random() ** 2
        _y2 = r.random() ** 2
        if m.sqrt(_x2 + _y2) < 1.0:
            _inside = _inside + 1
    _pi = float(_inside) / _total * 4
    print('Esitmated value of Pi: {} '.format(_pi))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## VI. HPC Architecture
        As mentioned earlier, an HPC system is generally not just one machine, instead it is a set of individual computers networked together. This network is similar in concept to the Internet, though it is usually much faster and the network topology (the way the computers are organized in relation to each other) is different. In the case of the cluster this course is running on, the network is a bus topology, meaning that all of the nodes are plugged into a single network switch. The graphic below illustrates some common network topologies:
        ![Network Topologies](https://techiereader.com/wp-content/uploads/2016/08/Network-Topology.png)

        With multiple computers working together, HPC systems are able to achieve massive performance. The fastest system in the world is a cluster like this which runs at a peak of roughly 1100 Petaflops, or 1100 quadrillion floating point operations per second. Because these systems have programs that take up more than a whole node, they need a way to understand which parts of programs are running on which nodes. The way this is usually handled is through inter-process communication. We will get into this concept more in topic "distributed algorithms", but essentially, one part of a program will send a message to another part of the same program and tell it what information it needs to know in order to do its job. The following image describes this process (MPI message passing):
        ![Message Passing Diagram](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/coll-fig1.gif)

        In the image labeled "broadcast", a message-passing task is sending a message to all other running tasks. This can be useful to orchestrate many copies of the same program. In the image labeled "scatter", the task sends a personalized message to each other running task. this is useful when the same instruction needs to be performed with different input data. In the images labeled "gather" and "reduction," the task receives messages from all other tasks and creates a summary of all the input data. This can be used in situations where data is generated by one part of a workflow and needs to be processed later. Please keep in mind that there are many more applications of inter-process communication. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1.6 - Pinging Other Nodes 
        "You're not alone out there"

        As an example of a task involving many nodes, we're going to check around in our network and see if any other compute nodes are accessible or available. To do this, we're going to use Python's `subprocess` module to interact with the operating system.
        """
    )
    return


@app.function
# Defining ping function

def ping(host):
    """
    Returns True if host responds to a ping request
    """
    import subprocess, platform

    # Ping parameters as function of OS
    ping_str = "-n 1" if  platform.system().lower()=="windows" else "-c 1"
    args = "ping -i 0.2 " + " " + ping_str + " " + host

    # Ping
    return subprocess.call(args, shell=True) == 0


@app.cell
def _():
    # Ping all addresses in our 172.16.52.0/24 network and print ones that seem up
    for j in range(0, 256):
        host = "172.18.0.{}".format(j)
        res = ping(host)
        if res:
            print(host)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you waited long enough (and i don't blame you if you didn't) you will have seen that there are at least four machines up in this network, which could grow to a size of 255.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise 1.6 - Optional exercise

         Optional Exercise 1. Write a program to compute sums of _n_ integers in parallel,
    
        Using your new HPC skills, write a short program that generates two random vectors of length _n_, sums them in parallel, and prints out how long it took for them to be added.
   
   
        """
    )
    return


@app.cell
def _():
    # your code goes here
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

