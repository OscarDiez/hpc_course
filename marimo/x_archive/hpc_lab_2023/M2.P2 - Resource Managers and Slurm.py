import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # M2.P1 OS, Resource Managers & Slurm
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Interacting With the Operating System
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
        ## 2.Resource managers
        ## 2.1 What is a Resource Manager?
        You will recall that we defined an HPC system as one made up of smaller constituent systems all working together. However, until now, all of the code we have written has run on just this one computer, which is the login node of a cluster. This is because we have not yet learned to use a _resource manager_. A _resource manager_ is a program that contains both a server, running on a head node, and any number of clients, running on worker nodes. The client allows worker nodes to ask the head node for work, and the server provides jobs to carry out. Almost all clusters have some form of resource manager on them which allows users to submit and monitor jobs to be run on the worker nodes. Most resource managers also have scheduling systems which allow them to run jobs in different orders based on a number of parameters. The following image describes the job flow of Slurm, a commonly used resource manager:

        ![SLURM architecture](https://slurm.schedmd.com/arch.gif)


        Users submit tasks to a queue, which are then ordered by priority rules set by administrators, and those jobs get run on any available backend resources.


        **srun** is used to submit a job for execution in real time

        while

        **sbatch** is used to submit a job script for later execution.

        They both accept practically the same set of parameters. The main difference is that srun is interactive and blocking (you get the result in your terminal and you cannot write other commands until it is finished), while sbatch is batch processing and non-blocking (results are written to a file and you can submit other commands right away).

        If you use **srun** in the background with the & sign, then you remove the 'blocking' feature of srun, which becomes interactive but non-blocking. It is still interactive though, meaning that the output will clutter your terminal, and the srun processes are linked to your terminal. If you disconnect, you will loose control over them, or they might be killed (depending on whether they use stdout or not basically). And they will be killed if the machine to which you connect to submit jobs is rebooted.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2.2 - Submitting and Monitoring a Simple Resource Manager Job
        The resource manager we have installed on this system is called Simple Linux Utility for Resource Management, commonly abbreviated to Slurm. In this example, we will use `os.popen()`  to submit a simple job to the resource manager and monitor it.

        The SGE interface is based on the `srun` and `squeue` commands. `srun` allows users to submit  shell scripts and `squeue` is used for monitoring jobs.
        """
    )
    return


app._unparsable_cell(
    r"""
    # Interacting with the resource manager

    import subprocess
    import time

    path = \"data/submit_script.sh\"
    outpath = \"data/job_out.out\"

    #os.environ['SGE_ROOT'] = '/local/cluster/sge'
    if os.path.exists(path):
        os.remove(path)
    
    if os.path.exists(outpath):
        os.remove(outpath)


    # Write submit scripts
    with open(path,\"w\") as file:
        file.write(\"#!/bin/bash\n\")
        file.write(\"echo h3ll0! >> {}\n\".format(outpath))
        file.write(\"sleep 5\n\")
        file.write(\"echo h3ll0! >> {}\n\".format(outpath))

    # Submit job
    !sbatch ./data/submit_script.sh -e /dev/null

    # List running jobs
    !squeue
    """,
    name="_"
)


@app.cell
def _(os, outpath):
    if os.path.exists(outpath):
        with open(outpath) as file_1:
            print(file_1.read())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2.2 Programmatic Interaction
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
    # Submit lots of Slurm jobs all at once

    #manyout = \"/data/manyjob_out.out\"
    #if os.path.exists(manyout):
    #    os.remove(manyout)

    for i in range(1,11):
        !srun hostname
    """,
    name="_"
)


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
        # 3. Slurm Cluster
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3.1 Introduction to Slurm

        ![image.png](attachment:e41a398e-040d-41d7-8e81-c7e6a3d4eea2.png)

        According to the definition, Slurm is an open-source, fault-tolerance, highly scalable cluster management and job scheduling system for all sizes of Linux clusters.

        Slurm stands Simple Linux Utility for Resource Management (SLURM), and it is used by many of the world’s supercomputers and Linux clusters in general. In simple words, Slurm allows us to execute jobs in the cluster using the resources of nodes that are part of it.

        Slurm Architecture
        You are going to use a Slurm cluster using docker-compose, that allows us to create an environment from docker images (previously built). Docker-compose will create containers and the network to communicate them in an isolated environment. Each container will be a component of the cluster.

        slurmmaster is the container with slurmctld (The central management daemon of Slurm).

        slurmnode[1–3] are the containers with slurmd (The compute node daemon for Slurm).

        slurmjupyter is the container with jupyterlab. This allows interacting with the cluster using JupyterLab as a cluster client. As end-users, we’ll work using JupyterLab through a browser for interacting with Slurm.

        cluster_default network, docker-compose will create a network to join and keep all containers altogether. Containers inside the network can see each other.

        The following diagram shows how all components interact

        ![image1.png](attachment:0287f884-b835-4051-ab02-c5a577f2cf31.png)

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## II. Using Slurm 

        Now, in your browser go to the following URL to access to JupyterLab.

        http://localhost:8888
        You’ll see the JupyterLab environment.

        ![image5.png](attachment:b3cfdbba-dbca-460a-9ec6-11f85bca6185.png)

        JupyterLab with Slurm Queue and Client
        It’s installed HPC Tools / Slurm Queue extension.

        ![image6.png](attachment:e01ece8c-54b6-4694-b6d9-a7003f37c4a8.png)

        Push the button, and you’ll get the Slurm Queue Manager.

        ![image7.png](attachment:73e4dff6-d0e0-4e14-8533-423962fcf261.png)

        To get a sight of the cluster, go to a Terminal in the Launcher tab. You can use the terminal to launch slurm jobs via commands.

        ![image8.png](attachment:1cc74176-2955-4a1e-9f85-d501faa6dff4.png)

        In the Terminal, execute the command:

                $ scontrol show node 

        This command provides you with information about the cluster and the nodes.
        The cluster creates a Partition called slurmpar. Partitions in Slurm are sets of nodes with associated resources.
        You will see the number of nodes/sockets, memory and resouce consumption in the cluster, and the resources of each node. 


        ### Example 2.4 - getting the configuration of the nodes
        """
    )
    return


@app.cell
def _(subprocess):
    subprocess.run('scontrol show node', shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## III. Launching jobs  

        ### Example 2.5 - creating and executing jobs using files and slurm queue manager
        We are going to develop an example using python to test how to work our cluster.

        Go to JupyterLab and create a New File and Rename the File as **test.py**

        ![image9.png](attachment:6c7f5fbf-ea5b-44d0-b94f-8d4b6b533cef.png)

        And write the following code:

        `#!/usr/bin/env python3
        import socket
        from datetime import datetime as dt
        if __name__ == '__main__':
            print('Process started {}'.format(dt.now()))
            print('NODE : {}'.format(socket.gethostname()))
            print('PID  : {}'.format(os.getpid()))
            print('Executing for 15 secs')
            time.sleep(15)
            print('Process finished {}\n'.format(dt.now()))`
    

        Now, we’ll write a job.sh script. Go to New File again and rename the File to **job.sh**, write the following:

        `#!/bin/bash
        #
        #SBATCH --job-name=test
        #SBATCH --output=result.out
        #
        #SBATCH --ntasks=6
        #
        sbcast -f test.py /tmp/test.py
        srun python3 /tmp/test.py` 


        In the script, I define the output file result.out, and ntask=6 because we have 3 nodes with 2 CPUs each.

        **sbcast** will transmit the file to the nodes allocated to a Slurm job.

        **srun** will run parallel jobs.

        Our test.py will be executed in parallel as 6 tasks.

        Go to Submit Job in Slurm Queue Manager, and choose job.sh (path /home/admin).


        ![image10.png](attachment:2ed23ae7-a174-43aa-9651-287e12104806.png)

        After executes the job.sh script, push Reload button, you’ll see the following.

        ![image11.png](attachment:fc4c78b6-5be6-4c5d-8ddd-f0cd585e9218.png)


        After 15 secs, the results will be written in the file result.out.


        `Process started 2021-02-28 11:23:55.094187
        NODE : slurmnode1
        PID  : 249
        Executing for 15 secs
        Process finished 2021-02-28 11:24:10.109268
        Process started 2021-02-28 11:23:55.133633
        NODE : slurmnode3
        PID  : 145
        Executing for 15 secs
        Process finished 2021-02-28 11:24:10.141112
        Process started 2021-02-28 11:23:55.149958
        NODE : slurmnode3
        PID  : 144
        Executing for 15 secs
        Process finished 2021-02-28 11:24:10.164342
        Process started 2021-02-28 11:23:55.153752
        NODE : slurmnode1
        PID  : 248
        Executing for 15 secs
        Process finished 2021-02-28 11:24:10.168402
        Process started 2021-02-28 11:23:55.192345
        NODE : slurmnode2
        PID  : 145
        Executing for 15 secs
        Process finished 2021-02-28 11:24:10.207377
        Process started 2021-02-28 11:23:55.197817
        NODE : slurmnode2
        PID  : 146
        Executing for 15 secs
        Process finished 2021-02-28 11:24:10.212361`


        Analyzing the result, we can see that test.py was executed in parallel 6 times, starting and finishing at the same time (all tasks were executed in 15 secs), two times in every node.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## IV. Using command line in Slurm

        All these exercises can be run from here, the terminal or the Jupyter console. 
        Once you have run them here (notebook), try to run them from the terminal or even launch the jobs from the sLurm queue manager (jupyter extension). 

        To get a sight of the cluster, go to a Terminal in the Launcher tab. You can use the terminal to launch slurm jobs via commands.

        We will use the **sbatch** and **srun** command to run jobs. 

        We will run the command sinfo -N to get the status of the nodes:

        ### Example 2.6 - Execute slurm commands from terminal

        Here we are using  %% to execute commands from jupyter in the terminal. You can runt he same commands directly in the terminal
        """
    )
    return


@app.cell
def _(subprocess):
    subprocess.run('sinfo -N', shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        The 7th column of the output of sinfo -N -l will tell you how much memory is installed in each compute node.


        """
    )
    return


@app.cell
def _(subprocess):
    subprocess.run('sinfo -N -l', shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        And for the information about the partition: *scontrol show partition*

        """
    )
    return


@app.cell
def _(subprocess):
    subprocess.run('scontrol show partition', shell=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        To list all current jobs in the clustuer you can use squeue

        """
    )
    return


@app.cell
def _(subprocess):
    subprocess.run('squeue', shell=True)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

