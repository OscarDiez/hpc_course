import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Module 1. Practice 1
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Introduction to HPC Clusters and SLURM

        This notebook is designed to help you understand what an HPC (High-Performance Computing) cluster is, how to use SLURM for job scheduling, and how to compile and run a simple C program on the cluster.

        ## Learning Objectives
        - Understand the architecture of HPC clusters.
        - Learn the basics of SLURM and its main commands.
        - Compile and run a simple C program using SLURM.
        - Perform practical exercises to reinforce the learned concepts.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What is an HPC Cluster?
        An HPC cluster is a collection of interconnected computers (or nodes) that work together to perform complex computations. These clusters can handle computational tasks that require a lot of processing power and memory, far beyond what a single machine could manage.\

        ![Meluxina HPC Architecture](https://hpc.uni.lu/old/images/overview/meluxina_overview.png)


        ### Architecture of HPC Clusters
        1. **Management Node**: Controls the overall operation of the cluster.
        2. **Login Node**: Provides an interface for users to submit jobs and interact with the cluster.
        3. **Compute Nodes**: Perform the actual computations.


        """
    )
    return


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


        - **Scenario 2: Designing a New Car or Plane:** You're a brand new aerospace engineer working for the Mercedes-Benz Formula One team. You have the off season (usually between December and May, or about five months) to design a new car which is better than all the cars that beat you last year. Traditionally, the way to do this is start with a small model, put it in a wind tunnel, evaluate it, and repeat this process. Then, you slowly scale up to bigger models and eventually start building concept cars. However, you only have five months, and each model may take a month to design and produce. You simply don't have time. Instead, you get started with your HPC system and start creating some [Computational Fluid Dynamics](https://en.wikipedia.org/wiki/Computational_fluid_dynamics) models which you can then use to create your new car with plenty of time to spare. The image below is the output of a CFD model. 


        ![CFD model of car](https://upload.wikimedia.org/wikipedia/commons/f/fa/Verus_Engineering_Porsche_987.2_Ventus_2_Package.png)


        - **Scenario 3: Personalized Medicine and Drug Discovery:** Life sciences are another major vertical segment that relies on HPC technologies in various application areas. Supercomputing is used by researchers and enterprises for genome sequencing and drug discovery. Pharmaceutical companies often deploy supercomputers to accelerate the process of drug discovery using various molecular dynamic simulation methodologies. Using HPC and molecular dynamics simulations researchers are able to design new drugs and virtually test effectiveness, enabling significant optimization of the research process while resulting in safer and more effective drugs. HPC is also used to develop virtual models of human physiology (e.g., heart, brain, etc.), which enable scientists and researchers to understand ailments and potential treatments better. Increasingly life sciences researchers and companies are engineering new methodologies combining genome sequencing and drug discovery to enable new and more effective forms of personalized medicine that could cure some of the most challenging diseases.


        ![computational climate research](https://www.cbkscicon.com/wp-content/uploads/2019/09/small_crop_Screen-Shot-2018-03-08-at-17.17.33-1-300x300.png)
        """
    )
    return


@app.cell
def _():
    # Enable oversubcription in the cluster
    import os
    os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    def run_srun(command):
        os.system(f"srun --oversubscribe {command}")
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you want to change the code you can do it in the filesystem. To access from Jupyter you can do it from the left panel. Select your file inside folder openmp and modify it (do not forget to save it with File/save). 
        Do not forget that you will need to compile it. You can reuse the previous code above or run the same commands directlyl in the terminal.

        You can launch a terminal directly from jupyter launcher or just from docker (if from docker you can sue the `bash` command from docker to get a proper bash terminal). 

        To do it from jupyter, start a Terminal in the Launcher tab. You can use the terminal to launchany command, including slurm jobs via commands.


        # Compiling and Running Programs on an HPC System

        This notebook will guide you through the steps necessary to compile and run a computationally intensive C program on a High-Performance Computing (HPC) system. We will cover both basic and advanced topics, focusing on using specific compilers and modules available on the HPC.

        ## Why Use an HPC for Compiling?

        Compiling and running programs on an HPC system can significantly enhance performance for compute-intensive tasks. This is due to several advantages that HPC systems provide:
        - **Access to specialized compilers and libraries:** Optimized to exploit the hardware capabilities like multiple cores, high-performance GPUs, and fast interconnects.
        - **Module systems for easy software management:** Allows users to easily load and switch between different software environments and libraries needed for different applications.
        - **Enhanced computational power:** With more processors, memory, and storage than a typical desktop or laptop, HPC systems can handle much larger computations.

        ## Example Program: `calculate_pi.c`

        Instead of a simple hello world program, we will use a more complex C program that calculates the value of Pi using the Monte Carlo method. This method involves simulating random points and assessing how many fall within a quarter circle inscribed in a unit square. The ratio of points inside the circle to the total points approximates Pi/4.

        Here's the source code for `calculate_pi.c`:

        """
    )
    return


@app.cell
def _(os):
    # Define the path for the C program file
    c_program_path = "calculate_pi.c"

    # Remove the existing file if it exists
    if os.path.exists(c_program_path):
        os.remove(c_program_path)

    # Create and write the C program
    c_program = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>

    int main(int argc, char *argv[]) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <iterations>\\n", argv[0]);
            return 1;
        }
    
        int iterations = atoi(argv[1]);
        if (iterations <= 0) {
            fprintf(stderr, "Please provide a positive integer for iterations.\\n");
            return 1;
        }

        int inside = 0;
        double x, y, pi;

        srand(time(NULL)); // Seed the random number generator

        for (int i = 0; i < iterations; i++) {
            x = (double)rand() / RAND_MAX;
            y = (double)rand() / RAND_MAX;
            if (x * x + y * y <= 1) {
                inside++;
            }
        }

        pi = (double)inside / iterations * 4;
        printf("Approximation of Pi: %f\\n", pi);

        return 0;
    }
    """

    # Write the C program to a file
    with open(c_program_path, "w") as file:
        file.write(c_program)

    print(f"Complex C program written to {c_program_path} with command-line argument support.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 2. Compile the Program

        Use the `gcc` command to compile `calculate_pi.c` and generate an executable named `calculate_pi`:

        1. Load the Necessary Modules
        HPC systems use module systems to manage software environments. Before compiling, load the appropriate compiler module. Here, we'll use the GCC compiler:
        """
    )
    return


@app.cell
def _(os):
    import subprocess

    # Compile the C program using gcc
    compile_command = "gcc calculate_pi.c -o calculate_pi"  # Corrected output file name
    compile_process = subprocess.run(compile_command, shell=True, capture_output=True, text=True)

    # Print the output and error (if any) after compilation attempt
    print("Compiling the C program...")
    if compile_process.stdout:
        print("Output:", compile_process.stdout)
    if compile_process.stderr:
        print("Error:", compile_process.stderr)

    # Check if the executable was created
    if os.path.exists("calculate_pi"):  # Corrected executable file name
        print("Compilation successful, executable 'calculate_pi' created.")
    else:
        print("Compilation failed.")
    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Run the Program

        Now we will execute the program. 

        **As it is doing  100000000 ITERATIONS it will take time, Be patient!** 

        Execute the program with the following command to see the output:

        """
    )
    return


@app.cell
def _(subprocess):
    compile_command_1 = ['gcc', 'calculate_pi.c', '-o', 'calculate_pi']
    subprocess.run(compile_command_1)
    run_program = subprocess.run(['./calculate_pi', '100000000'], capture_output=True, text=True)
    print(run_program.stdout)
    print(run_program.stderr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Resource managers and Slurm
        ## What is a Resource Manager?
        An HPC system is made up of smaller constituent systems all working together. Normally, all of our interactions  are with one computer, which is the login node of a cluster. This is because we have not yet learned to use a _resource manager_. A _resource manager_ is a program that contains both a server, running on a head node, and any number of clients, running on worker nodes. The client allows worker nodes to ask the head node for work, and the server provides jobs to carry out. Almost all clusters have some form of resource manager on them which allows users to submit and monitor jobs to be run on the worker nodes. Most resource managers also have scheduling systems which allow them to run jobs in different orders based on a number of parameters. 



        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to SLURM

        SLURM (Simple Linux Utility for Resource Management) is a powerful scheduler that helps manage resources and schedule jobs on an HPC cluster.

        The following image describes the job flow of Slurm, a commonly used resource manager:

        ![SLURM architecture](https://slurm.schedmd.com/arch.gif)

        ### Main SLURM Commands
        - `srun`: Run parallel jobs.
        - `sbatch`: Submit a batch job script to the scheduler.
        - `squeue`: View the job queue.
        - `scancel`: Cancel a job.
        - `sinfo`: View information about the nodes and partitions.

        In this notebook, we will create, compile, and run a simple C program using SLURM.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding Cluster Configuration with `sinfo`

        The `sinfo` command in SLURM provides detailed information about the current state of the nodes and partitions within the HPC cluster. This command is essential for users to understand the availability and status of resources before submitting jobs.

        ## Key Outputs of `sinfo`

        - **PARTITION**: Shows the partition names.
        - **AVAIL**: Indicates if the partition is available (`up`) or not (`down`).
        - **TIMELIMIT**: Lists the maximum time that jobs are allowed to run in the partition.
        - **NODES**: Shows the number of nodes in each state.
        - **STATE**: Indicates the state of the nodes (e.g., `idle`, `alloc` for allocated, etc.).
        - **NODELIST**: Provides the specific names or identifiers of the nodes.

        By default, `sinfo` displays a brief summary. To get more detailed information, you can use various flags with this command.

        ## Example Commands

        - `sinfo`: Provides a basic overview of the cluster.
        - `sinfo -l`: Provides a detailed view.
        - `sinfo -N`: Lists information node by node.
        - `sinfo -s`: Displays a short format.

        Let's run a basic `sinfo` command to see the current state of the cluster.

        """
    )
    return


app._unparsable_cell(
    r"""
    !sinfo
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding Cluster Control with `scontrol`
        The `scontrol` command in SLURM is a powerful tool used to display and modify the configuration and state of various cluster components, such as nodes, partitions, jobs, and more. This command is especially useful for administrators or advanced users who need to view detailed cluster information or modify resource settings.

        ### Key Functionalities of `scontrol`
        - **Show Node Information:** Displays detailed information about individual nodes in the cluster, such as their CPU count, memory, state, and more.
        - **Show Partition Information:** Retrieves detailed information about the partitions in the cluster, including their resource limits and node assignments.
        - **Show Job Information:** Displays information about specific jobs, including their current state, resources used, and associated nodes.
        - **Modify Jobs/Nodes:** Allows administrators to update the state of nodes or jobs (e.g., draining a node or cancelling a job).

        ### Example Commands

        #### 1. **Display Partition Information**
        ```bash
        scontrol show partition

        """
    )
    return


app._unparsable_cell(
    r"""
    !scontrol show partition
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    !scontrol show node
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Creating and Submitting a SLURM Job

        Users submit tasks to a queue, which are then ordered by priority rules set by administrators, and those jobs get run on any available backend resources.


        **srun** is used to submit a job for execution in real time

        while

        **sbatch** is used to submit a job script for later execution.

        They both accept practically the same set of parameters. The main difference is that srun is interactive and blocking (you get the result in your terminal and you cannot write other commands until it is finished), while sbatch is batch processing and non-blocking (results are written to a file and you can submit other commands right away).

        If you use **srun** in the background with the & sign, then you remove the 'blocking' feature of srun, which becomes interactive but non-blocking. It is still interactive though, meaning that the output will clutter your terminal, and the srun processes are linked to your terminal. If you disconnect, you will loose control over them, or they might be killed (depending on whether they use stdout or not basically). And they will be killed if the machine to which you connect to submit jobs is rebooted.
        To run our compiled program on the HPC cluster, we need to create a SLURM job script. This script specifies the resources required and the commands to execute.

        ### SLURM Job Script Example
        Below is a simple SLURM script that requests 1 compute node for 5 minutes and runs our `hello_hpc` executable.

        ```bash
        #!/bin/bash
        #SBATCH --job-name=calculate_pi
        #SBATCH --output=calculate_pi.out
        #SBATCH --error=calculate_pi.err
        #SBATCH --time=00:05:00
        #SBATCH --nodes=1
        #SBATCH --mem=1G  # Allocates 1 GB of total memory to the job

        # Load necessary modules
        module load gcc

        # Run the executable
        srun ./calculate_pi 1000000000

        """
    )
    return


@app.cell
def _(os):
    slurm_script_path = 'calculate_pi.slurm'
    if os.path.exists(slurm_script_path):
        os.remove(slurm_script_path)
    slurm_script = '#!/bin/bash\n#SBATCH --job-name=calculate_pi\n#SBATCH --output=calculate_pi.out\n#SBATCH --error=calculate_pi.err\n#SBATCH --time=00:05:00\n#SBATCH --nodes=1\n#SBATCH --mem=500M  # Allocates 500MB of total memory to the job\n\n# Load necessary modules\nmodule load gcc\n\n# Run the executable\nsrun ./calculate_pi 1000000000\n'
    with open(slurm_script_path, 'w') as file_1:
        file_1.write(slurm_script)
    print(f'SLURM job script written to {slurm_script_path}.')
    os.chmod(slurm_script_path, 493)
    with open(slurm_script_path, 'r') as file_1:
        script_content = file_1.read()
    print('\nContents of the SLURM job script:')
    print('----------------------------------')
    print(script_content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Submitting and Monitoring a SLURM Job in Jupyter Notebook

        This section of the notebook demonstrates how to submit a SLURM job using the `sbatch` command and monitor its status using the `squeue` command. We will execute these commands directly from the Jupyter Notebook using the `!` syntax, which allows us to run shell commands in a more interactive manner.

        #### Submitting the SLURM Job

        We use the `sbatch` command to submit a job to the SLURM scheduler. The job script `calculate_pi.slurm` contains instructions for the SLURM workload manager on how to execute the task. This script specifies the resources needed and the executable to run.

        #### Allowing Time for Job Queueing
        To ensure that the job is queued before we check its status, we include a short delay using Python's time.sleep() function. This is crucial as SLURM may take a few moments to update the queue, especially in busy environments.

        #### Checking the Job Status
        After submitting the job, we use the squeue command to check the status of jobs in the queue. This command lists all jobs that are currently queued or running, allowing us to monitor the status of our job.

        """
    )
    return


app._unparsable_cell(
    r"""
    import time

    # Submit the SLURM job using the `!` syntax for direct shell command execution
    !sbatch {\"calculate_pi.slurm\"}

    # Wait for a few seconds to ensure the job is queued
    time.sleep(3)

    # Check the status of the job queue
    !squeue
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Wait until the job has finished. You can check it running multiple times the command below:
        """
    )
    return


app._unparsable_cell(
    r"""
    !squeue
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Examining SLURM Job Output and Error Files

        Once a SLURM job is submitted and executed, it generates output and error files specified in the job script. These files contain valuable information about the execution of the program, including any results printed to the console and any error messages that occurred during execution.

        #### Understanding Output and Error Files

        ##### Output File (`calculate_pi.out`) **

        The output file contains the standard output from the program execution. This includes any `printf` statements or other console outputs generated by the C program. In our case, this file will contain the approximate value of Pi calculated by our program.

        ##### Error File (`calculate_pi.err`)

        The error file captures any standard error messages produced by the program. This includes any compilation or runtime errors, warnings, or other messages that are sent to the error stream.

        #### Code to Display the Contents of Output and Error Files

        Let's write code to read and display the contents of these files, allowing us to verify the results and diagnose any potential issues.

        """
    )
    return


@app.cell
def _(os):
    output_file = 'calculate_pi.out'
    error_file = 'calculate_pi.err'
    if os.path.exists(output_file):
        print(f'\nContents of {output_file}:')
        print('----------------------------------')
        with open(output_file, 'r') as file_2:
            output_content = file_2.read()
            print(output_content)
    else:
        print(f'\n{output_file} does not exist.')
    if os.path.exists(error_file):
        print(f'\nContents of {error_file}:')
        print('----------------------------------')
        with open(error_file, 'r') as file_2:
            error_content = file_2.read()
            print(error_content)
    else:
        print(f'\n{error_file} does not exist.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding `srun` in SLURM

        In SLURM, both `sbatch` and `srun` are used to execute jobs on an HPC cluster, but they serve different purposes and are used in distinct scenarios. Understanding when to use each command is essential for effective job management and resource utilization.

        ## `sbatch` vs. `srun`

        ### `sbatch`

        - **Purpose**: Submits a batch job script to the scheduler to be executed at a later time when resources become available.
        - **Usage**: Primarily used for batch processing of non-interactive tasks, where you write a script with job specifications and submit it to the queue.
        - **Execution**: The job runs according to the specified resources and constraints in the SLURM script without user interaction during execution.

        ### `srun`

        - **Purpose**: Launches parallel tasks and can be used for both interactive and non-interactive job execution.
        - **Usage**: Often used for interactive jobs or to launch parallel tasks within an already scheduled job.
        - **Execution**: `srun` can be used to run tasks interactively on compute nodes or to start tasks within a running job environment, providing more flexibility for dynamic task execution.

        ## When to Use `srun`

        - **Interactive Jobs**: Use `srun` to start an interactive session on a compute node for testing, debugging, or running tasks interactively.
        - **Within Scripts**: Use `srun` within an `sbatch` script to launch parallel tasks that require coordination across multiple CPUs or nodes.
        - **Dynamic Execution**: Use `srun` to dynamically allocate resources and run tasks without needing to pre-write a batch script.

        ## Example Usage

        We will demonstrate how to use `srun` to run a simple interactive job and a parallel computation task.


        """
    )
    return


app._unparsable_cell(
    r"""
    # Use srun to start an interactive session on a compute node
    # Note: This command is typically run in a terminal, not directly executable in a Jupyter Notebook.

    !srun hostname

    # Explanation:
    # --pty: Allocates a pseudo-terminal from the compute node allocated, allowing interactive command execution.
    # bash -i: Starts an interactive bash shell session.
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interactive SLURM Usage in Jupyter Terminal

        This guide will help you explore SLURM commands interactively within a Jupyter terminal. By practicing these commands, you'll gain familiarity with job scheduling, monitoring, and resource management on an HPC cluster.

        ### 1. Access the Shell in Jupyter

        #### Open a New Terminal

        - **Open a Launcher**: Click on the `+` icon or `New Launcher` to open the launcher.
        - **Select Terminal**: From the launcher, click on `Terminal` to open a new shell session. This terminal acts like a login node interface.

        ### 2. Run Basic Linux Commands

        Before diving into SLURM, familiarize yourself with some basic Linux commands to navigate and manage your files.

        - **List Files and Directories**: 
            - Run the command `ls` to show the content of the current folder.
  
        - **Print Current Directory**:
            - Run the command `pwd` to display the current directory path.

        ### 3. SLURM Commands for Job Management

        Learn how to interact with SLURM to manage and monitor your computational jobs.

        - **Check Available Partitions**:
          - Run the command `sinfo` to display available partitions and their status. This is useful for determining resource availability and node types.

        - **View Job Queue**:
          - Run the command `squeue` to show the current job queue. This command displays jobs currently running or waiting, along with their IDs, user names, and statuses.

        - **Submit a Job Script**:
          - Use the command `sbatch calculate_pi.slurm` to submit a batch job to the SLURM scheduler for execution when resources are available. Replace `calculate_pi.slurm` with the name of your actual job script.

        - **Check Your Job Status**:
          - Use `squeue -u $USER` to list all jobs submitted by the current user, allowing you to monitor their progress and status.

        - **Cancel a Job**:
          - Run `scancel <job_id>` to cancel a job specified by its job ID. Replace `<job_id>` with the actual job ID you wish to cancel.

        ### 4. Running Interactive Jobs

        Explore interactive job sessions to dynamically test and run tasks on compute nodes.

        - **Start an Interactive Session**:
          - Use `srun --pty bash -i` to allocate resources and start an interactive bash session on a compute node. This is ideal for debugging and interactive computations.

          **What You Can Do**:
          - Run commands interactively.
          - Test scripts with immediate feedback.
          - Explore resource usage in real-time.

        ### 5. Analyze Job Performance with `sacct`

        After jobs have completed, use `sacct` to gather detailed information about their execution.

        - **View Completed Job Details**:
          - Run `sacct --format=JobID,JobName,User,State,Elapsed,CPUTime,MaxRSS` to provide detailed statistics for completed jobs, such as CPU time, memory usage, and job state. This helps in understanding job performance and resource utilization.

        ## Discussion and Reflection

        - **Efficiency**: Reflect on how interactive SLURM commands enhance your ability to manage computational workloads effectively.
        - **Troubleshooting**: Consider how interactive sessions can assist in diagnosing job issues and refining scripts.
        - **Further Exploration**: Explore additional SLURM commands and options to optimize job scheduling and resource allocation.

        By following this guide, you will gain hands-on experience with SLURM and Linux shell commands, equipping you with the skills needed to navigate and utilize HPC resources effectively.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Understanding `sacct` in SLURM

        The `sacct` command in SLURM is used to report accounting information about jobs and job steps that are managed by the SLURM workload manager. It provides detailed information about the jobs, such as resource usage, runtime statistics, and job states, which are crucial for performance analysis and optimization.

        ### Key Features of `sacct`

        - **Job and Step Information**: `sacct` provides data on both jobs and individual job steps, offering insights into how resources were utilized at each stage of execution.
        - **Comprehensive Metrics**: Reports on CPU time, memory usage, job states, exit codes, and more, helping users identify bottlenecks or inefficiencies.
        - **Historical Data**: Accesses records of past jobs, allowing users to review previous job performances and resource consumption.

        ### Common `sacct` Options

        - `-j <job_id>`: Specifies a particular job ID to retrieve information for that job.
        - `--format`: Customizes the output format by specifying the fields to display.
        - `--starttime`: Limits the report to jobs that started after a specified time.
        - `-a` or `--allusers`: Displays information for all users (requires admin privileges).

        ### Example Usage

        In this environmnet we do not have activated the DB for the sacc so it is not possible to use it here. I include some examples:

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## HPC Job Submission with Multiple Nodes in SLURM

        ### Overview

        In high-performance computing (HPC) environments, jobs often need to be distributed across multiple nodes to fully utilize the available resources. SLURM (Simple Linux Utility for Resource Management) is a job scheduling system that efficiently manages the distribution of these jobs across a cluster. By submitting a job that utilizes multiple nodes, users can parallelize tasks and significantly reduce computation time.

        ### Running a Job on Multiple Nodes

        To run a job on multiple nodes using SLURM, you must create a SLURM batch script. This script specifies the resources your job requires, such as the number of nodes, CPUs per task, and the commands to be executed. When you submit this script to SLURM, the scheduler allocates the requested resources, executes the job, and handles output and error logging.

        ### What Happens When You Submit the Job

        1. **Resource Allocation:**
           - When the job is submitted via the `sbatch` command, SLURM schedules the job based on the requested resources and current availability within the cluster. If the script requests multiple nodes, SLURM will allocate the specified number of nodes.

        2. **Task Distribution:**
           - SLURM uses the `srun` command within the script to distribute tasks across the allocated nodes. Each node executes the specified command or program. For example, if the command is `hostname`, each node will execute it, and the hostname of each node will be returned.

        3. **Execution and Output:**
           - The tasks are executed in parallel on the different nodes. The output from these tasks is captured and saved to files specified in the script (e.g., `output.out` for standard output and `error.err` for errors). This allows you to review the results and any potential issues after the job has completed.

        4. **Completion and Monitoring:**
           - Once the tasks are completed, SLURM releases the resources, making them available for other jobs. You can monitor the job's progress using the `squeue` command, which shows the job status and other details.

        ### Explanation of the Process

        - **Parallel Execution:** By distributing the job across multiple nodes, SLURM allows tasks to be executed in parallel, leveraging the full computational power of the cluster. This is particularly beneficial for large-scale computations that would be time-prohibitive on a single node.

        - **Efficiency and Scalability:** SLURM efficiently manages resource allocation and job scheduling, ensuring that resources are not wasted and that the cluster operates at optimal efficiency. This scalability is key to handling the complex workloads typical in HPC environments.

        - **Output Management:** The output and error management features of SLURM make it easy to track and debug jobs. By directing output to specific files, users can review results and diagnose issues without interfering with ongoing tasks.

        This overview provides a clear understanding of how to utilize SLURM for submitting jobs across multiple nodes in an HPC environment and what happens during the execution of such jobs.

        """
    )
    return


@app.cell
def _(os):
    try:
        slurm_script_path_1 = 'simple_multi_task.slurm'
        if os.path.exists(slurm_script_path_1):
            os.remove(slurm_script_path_1)
        slurm_script_1 = '#!/bin/bash\n#SBATCH --job-name=simple_multi_task    # Job name\n#SBATCH --output=simple_multi_task.out  # Standard output\n#SBATCH --error=simple_multi_task.err   # Standard error\n#SBATCH --time=00:05:00                 # Time limit of 5 minutes\n#SBATCH --nodes=2                       # Number of nodes\n#SBATCH --ntasks-per-node=1             # Run one task per node\n#SBATCH --cpus-per-task=1               # Number of CPU cores per task\n#SBATCH --oversubscribe                 # Allow oversubscription\n#SBATCH --mem=1024M                     # Allocate 1GB of memory per node\n\n# Run hostname on each allocated node\nsrun /bin/hostname\n    '
        with open(slurm_script_path_1, 'w') as file_3:
            file_3.write(slurm_script_1)
        os.chmod(slurm_script_path_1, 493)
        with open(slurm_script_path_1, 'r') as file_3:
            script_content_1 = file_3.read()
        print('\nContents of the SLURM job script:')
        print('----------------------------------')
        print(script_content_1)
    except Exception as e:
        print(f'An error occurred: {e}')
    return


app._unparsable_cell(
    r"""
    # Submit the SLURM job using the `!` syntax for direct shell command execution
    !sbatch {\"simple_multi_task.slurm\"}

    # Wait for a few seconds to ensure the job is queued
    time.sleep(2)

    # Check the status of the job queue
    !squeue
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # Check the status of the job queue
    !squeue
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Wait until the job has finished or if there is a problem cancel it. 
        """
    )
    return


@app.cell
def _(os):
    output_file_1 = 'simple_multi_task.out'
    error_file_1 = 'simple_multi_task.err'
    if os.path.exists(output_file_1):
        print(f'\nContents of {output_file_1}:')
        print('----------------------------------')
        with open(output_file_1, 'r') as file_4:
            output_content_1 = file_4.read()
            print(output_content_1)
    else:
        print(f'\n{output_file_1} does not exist.')
    if os.path.exists(error_file_1):
        print(f'\nContents of {error_file_1}:')
        print('----------------------------------')
        with open(error_file_1, 'r') as file_4:
            error_content_1 = file_4.read()
            print(error_content_1)
    else:
        print(f'\n{error_file_1} does not exist.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using JupyterLab Terminal for HPC Practice

        Now that you've learned about HPC clusters and how to submit jobs using SLURM, it's time to practice these skills interactively. You can use the terminal in JupyterLab to mimic the process of working on a supercomputer, allowing you to gain hands-on experience with the commands and workflows you'll use in an actual HPC environment.

        ### 1. Access the JupyterLab Terminal

        - **Open a New Terminal**: In JupyterLab, click on the `+` icon or `File > New > Terminal` to open a new terminal window. This terminal session simulates a login node, where you can interact with the system as you would on an HPC cluster.

        ### 2. Basic Commands to Try

        Before diving into SLURM, familiarize yourself with a few basic Linux commands:
        - **List Files and Directories**: Use `ls` to display the contents of the current directory.
        - **Print Working Directory**: Use `pwd` to see the full path of the current directory.
        - **Create a Directory**: Use `mkdir test_directory` to create a new directory named `test_directory`.
        - **Change Directory**: Use `cd test_directory` to move into the newly created directory.

        ### 3. SLURM Commands to Practice

        With the terminal open, try the following SLURM commands to practice managing jobs:

        - **Check Available Partitions**: Run `sinfo` to see the available partitions and their statuses.
        - **Submit a Job Script**: Create a simple SLURM job script, then submit it using `sbatch script_name.slurm`. For example:
          ```bash
          #!/bin/bash
          #SBATCH --job-name=test_job
          #SBATCH --output=test_job.out
          #SBATCH --time=00:01:00
          #SBATCH --nodes=1
          #SBATCH --mem=500M

          echo "Running on $(hostname)"

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Using JupyterLab Terminal for HPC Practice

        Now that you've learned about HPC clusters and how to submit jobs using SLURM, it's time to practice these skills interactively. You can use the terminal in JupyterLab to mimic the process of working on a supercomputer, allowing you to gain hands-on experience with the commands and workflows you'll use in an actual HPC environment.

        ### 1. Access the JupyterLab Terminal

        - **Open a New Terminal**: In JupyterLab, click on the `+` icon or `File > New > Terminal` to open a new terminal window. This terminal session simulates a login node, where you can interact with the system as you would on an HPC cluster.

        ### 2. Basic Commands to Try

        Before diving into SLURM, familiarize yourself with a few basic Linux commands:
        - **List Files and Directories**: Use `ls` to display the contents of the current directory.
        - **Print Working Directory**: Use `pwd` to see the full path of the current directory.
        - **Create a Directory**: Use `mkdir test_directory` to create a new directory named `test_directory`.
        - **Change Directory**: Use `cd test_directory` to move into the newly created directory.
        - **View File Contents**: Use `cat` followed by a filename to view the contents of a file.

        ### 3. SLURM Commands to Practice

        With the terminal open, try the following SLURM commands to practice managing jobs:

        - **Check Available Partitions**: Run `sinfo` to see the available partitions and their statuses.
        - **Submit a Job Script**: Create a simple SLURM job script, then submit it using `sbatch script_name.slurm`. For example:
          ```bash
          #!/bin/bash
          #SBATCH --job-name=test_job
          #SBATCH --output=test_job.out
          #SBATCH --time=00:01:00
          #SBATCH --nodes=1
          #SBATCH --mem=500M

          echo "Running on $(hostname)"

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is the end of this part of the practice. 
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

