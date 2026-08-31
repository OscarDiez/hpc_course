import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # M2.P1 OS & Resource Managers
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## I. Interacting With the Operating System
        Programming languages are nice for carrying out computations based on input data. They have useful constructs like `for` loops, `if` statements, and module systems so that you can reuse code easily. However, for certain tasks, what's built in to the programming language is simply not enough. For situations like this, programming languages usually allow explicit interaction with the underlying operating system. In Python, this is handled by the `os` module. The following graphic helps to describe how the underlying OS behaves and how we can interact with it:


        ![OS Architecture Diagram](https://archive-docs.d2iq.com/mesosphere/dcos/2.1/img/architecture-layers-redesigned.png)

        The code you have been running in these notebooks fits into the "services" section of the software layer, and our OS interaction will allow us to request data and provide instructions to the objects in the "platform" layer.


        It is important to remember that Python code is meant to be portable, and you have to be careful when using the `os` module to ensure that your code can still be run on your target systems. This course is designed for \*nix  systems (including Linux, macOS and BSD) like the containers you are running now. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2.1a - Basic Filesystem Interaction
        In this example, we will interact with the filesystem around us. The module `os` contains a lot of useful functionality for interacting with the filesystem, networking layers, and other OS-specific methods. For more information see the [`os` docs](https://docs.python.org/3/library/os.html).
        """
    )
    return


@app.cell
def _():
    # Ex. 2.1a - Filesystem interaction 

    import os

    # We can list directories
    print(os.listdir('.'))

    # We can make directories
    print(os.mkdir('./dira'))

    # List again to see the new directory
    print(os.listdir('.'))

    # We can remove directories too
    print(os.rmdir('./dira'))

    # Finally, list again to see the new directory
    print(os.listdir('.'))


    # Using open(), we can read and write files
    with open("./data/message.txt", "w") as file:
        file.write("Super Secret Message")
    
    with open("./data/message.txt", "r") as file:
        print(file.read())
    
    # Delete our secret message
    os.remove("./data/message.txt")
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2.1b - Opening Processes
        For things as universal as filesystem interaction, `os` has built-in functions to help you out. However, there are some things that the Python developers simply can't prepare for. Let's say you want to compile some code, run it, and print the output, all from within Python. This is simple with the command line, and Python's `os` module provides a way to run system commands through the command line, through a function called `os.popen()`. Please remember that this can be unsafe if you allow malicious actors to access your computer, so only use the `os.popen()` function when necessary.
        """
    )
    return


@app.cell
def _(os):
    # Opening Processes
    # There is some code in /home/users/glick/intro-to-hpc/data/hello.c


    # Compile the code - There should be no output
    process = os.popen("cc ./data/hello.c -o ./data/hello.out ")
    print(process.read())

    # Run the code
    process = os.popen("./data/hello.out")
    print(process.read())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## II. What is a Resource Manager?
        You will recall that we defined an HPC system as one made up of smaller constituent systems all working together. However, until now, all of the code we have written has run on just this one computer, which is the login node of a cluster. This is because we have not yet learned to use a _resource manager_. A _resource manager_ is a program that contains both a server, running on a head node, and any number of clients, running on worker nodes. The client allows worker nodes to ask the head node for work, and the server provides jobs to carry out. Almost all clusters have some form of resource manager on them which allows users to submit and monitor jobs to be run on the worker nodes. Most resource managers also have scheduling systems which allow them to run jobs in different orders based on a number of parameters. The following image describes the job flow of Sun GridEngine, a commonly used resource manager: ![SGE architecture](https://2eof2j3oc7is20vt9q3g7tlo5xe-wpengine.netdna-ssl.com/wp-content/uploads/2015/10/Docker_Grid_Engine_R2.jpg)
        Users submit tasks to a queue, which are then ordered by priority rules set by administrators, and those jobs get run on any available backend resources.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2.2 - Submitting and Monitoring a Simple Resource Manager Job
        The resource manager we have installed on this system is called GridEngine, commonly abbreviated to SGE. In this example, we will use `os.popen()` and SGE to submit a simple job to the resource manager and monitor it.

        The SGE interface is based on the `qsub` and `qstat` commands. `qsub` allows users to submit arbitrary shell scripts and `qstat` is used for monitoring jobs.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Interacting with the resource manager

    import subprocess
    import time

    path = \"./data/submit_script.sh\"
    outpath = \"./data/job_out.out\"

    os.environ['SGE_ROOT'] = '/local/cluster/sge'
    os.remove(path)
    os.remove(outpath)


    # Write submit scripts
    with open(path,\"w\") as file:
        file.write(\"#!/bin/bash\n\")
        file.write(\"echo h3ll0! >> {}\n\".format(outpath))
        file.write(\"sleep 5\n\")
        file.write(\"echo h3ll0! >> {}\n\".format(outpath))

    # Submit job
    !qsub ./data/submit_script.sh -e /dev/null

    # List running jobs
    !qstat
    """,
    name="_"
)


@app.cell
def _(outputpath):
    with open(outputpath) as file_1:
        print(file_1.read())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## III. Programmatic Interaction
        Sometimes, you don't want to manually submit a job for each task you need to carry out. In this example, we will use a python loop to submit lots of jobs.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2.3 - Lots of Jobs!
        """
    )
    return


app._unparsable_cell(
    r"""
    # Submit lots of SGE jobs all at once

    os.environ['SGE_ROOT'] = '/local/cluster/sge'

    manyout = \"./data/manyjob_out.out\"
    os.remove(manyout)

    for i in range(1,11):
        !echo \"echo $i >> $manyout\" | qsub -e /dev/null
    """,
    name="_"
)


@app.cell
def _(manyout):
    with open(manyout) as file_2:
        print(file_2.read())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because this job is running in parallel, each number is not guaranteed to finish in the order in which it started. This is something that always needs to be considered when writing parallel algorithms.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## IV. Basic Dataflows
        Most of the time, a simple set of jobs running all at once is not enough for your purposes. Many times, you want to generate lots of different kinds of data all at once, and then perform some kind of summary statistics on it. Situations like this are often referred to as _dataflows_. Dataflows resemble real-world HPC workloads more accurately than the other examples we have looked at in the course so far. The following graphic represents a very simple dataflow to generate random numbers in parallel and take the average.
        ```
        App Calls    rand() rand() rand()
                      \     |     /
                       a    b    c
                        \   |   /
        App Call        avg_points()
                            |
                           avg
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2.4 - Array Job Plus Summary Statistics
        In this example, we will implement the dataflow described in the descriptive text above.
        We will use Python's `random` library to generate random numbers and print them to a file using SGE, then we will average all of them.
        """
    )
    return


app._unparsable_cell(
    r"""
    import random
    os.environ['SGE_ROOT'] = '/local/cluster/sge'

    # Submit lots of SGE jobs all at once

    rand_out = \"/home/users/glick/intro-to-hpc/data/rand_out.out\"
    os.remove(rand_out)

    # Generate random numbers
    for _ in range(1,11):
        !echo \"echo $((1 + RANDOM % 100)) >> $rand_out\" | qsub -e /dev/null
    """,
    name="_"
)


@app.cell
def _():
    rand_out = '/home/users/glick/intro-to-hpc/data/rand_out.out'
    with open(rand_out) as file_3:
        lines = [int(i) for i in file_3.readlines()]
        print(sum(lines) // len(lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercise 2. Array Dataflow with `qsub`
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

