import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercise 1: Exploring CPU and GPU Performance

        In this exercise, you will compare the performance of CPU and GPU in handling a computationally intensive task. We will use matrix multiplication as our benchmark, which is a common operation in scientific computing.

        ### Objectives
        - Understand the difference in computation time between a CPU and a GPU.
        - Learn how to use Google Colab's GPU for accelerating computations.

        ### Instructions
        0. Please select in colab an image with GPU enabled to run the code.
        1. Run the provided code to perform matrix multiplication using the CPU.
        2. Modify the code to utilize the GPU and observe the performance difference.
        3. Record the computation times for both CPU and GPU executions.


        ### Questions to Consider
        - How much faster is the GPU compared to the CPU for this task?
        - Why is there such a difference in performance?

        """
    )
    return


@app.cell
def _():
    import numpy as np
    import time
    import torch

    # Set the size of the matrix
    matrix_size = 1000

    # Generate random matrices
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)

    # Matrix multiplication on CPU
    start_time = time.time()
    C_cpu = np.dot(A, B)
    cpu_time = time.time() - start_time

    print(f"CPU computation time: {cpu_time:.4f} seconds")

    # Matrix multiplication on GPU using PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert matrices to tensors
    A_gpu = torch.tensor(A).to(device)
    B_gpu = torch.tensor(B).to(device)

    # Matrix multiplication on GPU
    start_time = time.time()
    C_gpu = torch.matmul(A_gpu, B_gpu)
    gpu_time = time.time() - start_time

    print(f"GPU computation time: {gpu_time:.4f} seconds")
    return (time,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercise 3: Simulating Job Scheduling with SLURM

        In this exercise, you will simulate job scheduling in an HPC environment. We'll implement two different scheduling strategies: First-Come-First-Serve (FCFS) and Priority-based scheduling. These examples will help you understand how job schedulers allocate resources and manage queues in HPC systems.

        ### Objectives
        - Simulate job scheduling and resource allocation using Python.
        - Compare the performance of First-Come-First-Serve (FCFS) and Priority-based scheduling.
        - Understand how `heapq.heappop` is used in priority-based scheduling.

        ### Instructions
        1. Run the code to simulate job scheduling using the First-Come-First-Serve (FCFS) strategy.
        2. Modify the code to include a priority-based scheduler.
        3. Add more job examples with varying execution times and priorities.
        4. Compare the efficiency of FCFS and priority-based schedulers by observing job completion times.

        ### Additional Examples
        - Implement a round-robin scheduler to see how it handles jobs differently.
        - Experiment with varying the job priorities and execution times to observe the changes in scheduling order and efficiency.

        ### Explanation of Priority-Based Scheduler and `heapq.heappop`

        The priority-based scheduler uses Python's `heapq` module to efficiently manage the job queue. In this implementation, jobs are stored in a heap, which is a complete binary tree that maintains the smallest element at the root. This property allows the scheduler to always select the highest-priority job (smallest value) quickly.

        - **`heapq.heappop`**: This function pops the smallest item from the heap, which in our case is the job with the highest priority. The heap is automatically adjusted after each pop to ensure that the smallest item is always at the root. This makes the priority-based scheduling both efficient and easy to manage.

        ### Questions to Consider
        - How does job priority affect the overall system efficiency?
        - What are the trade-offs between FCFS and priority-based scheduling?
        - How might different scheduling strategies impact the completion time of individual jobs?

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise 3: Simulating Job Scheduling with SLURM

        In this exercise, you will simulate job scheduling in an HPC environment using Python. We'll implement three different scheduling strategies: First-Come-First-Serve (FCFS), Priority-based scheduling, and Round-Robin scheduling. These examples will help you understand how job schedulers allocate resources and manage queues in HPC systems.

        ## Objectives

        - Simulate job scheduling and resource allocation using Python.
        - Compare the performance of First-Come-First-Serve (FCFS), Priority-based, and Round-Robin scheduling.
        - Understand how `heapq.heappop` is used in priority-based scheduling.

        ## Instructions

        1. Run the code to simulate job scheduling using the First-Come-First-Serve (FCFS) strategy.
        2. Compare the FCFS strategy with a priority-based scheduler.
        3. Experiment with the Round-Robin scheduler to see how it handles jobs differently.
        4. Add more job examples with varying execution times and priorities to observe the differences in scheduling behavior.

        Let's start with the implementation of the FCFS scheduler.

        """
    )
    return


@app.cell
def _(time):
    import heapq
    jobs = [(1, 'Job_1', 4, 2), (3, 'Job_2', 2, 1), (2, 'Job_3', 3, 3), (1, 'Job_4', 5, 2)]

    def _fcfs_scheduler(jobs):
        print('FCFS Scheduling')
        current_time = 0
        for job in jobs:
            print(f'Starting {job[1]} at time {current_time}')
            time.sleep(1)
            current_time = current_time + job[2]
            print(f'Completed {job[1]} at time {current_time}')
    print('=== FCFS ===')
    _fcfs_scheduler(jobs)
    return heapq, jobs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of FCFS Scheduling

        The **First-Come-First-Serve (FCFS)** scheduler is a straightforward strategy where jobs are processed in the order they arrive, regardless of their priority or resource requirements.

        - **Current Time**: The scheduler keeps track of the current time, which is updated as each job completes.
        - **Job Execution**: The jobs are executed one after the other based on their order in the queue. Each job's execution time is simulated with a short delay (`time.sleep(1)`).

        Now, let's implement and run a Priority-Based Scheduler to see how it differs from FCFS.

        """
    )
    return


@app.cell
def _(heapq, jobs, time):
    def _priority_scheduler(jobs):
        print('Priority-based Scheduling')
        heapq.heapify(jobs)
        current_time = 0
        while jobs:
            job = heapq.heappop(jobs)
            print(f'Starting {job[1]} at time {current_time}')
            time.sleep(1)
            current_time = current_time + job[2]
            print(f'Completed {job[1]} at time {current_time}')
    print('\n=== Priority-Based ===')
    _priority_scheduler(jobs[:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of Priority-Based Scheduling

        The **Priority-Based Scheduler** uses a priority queue to manage jobs, ensuring that the job with the highest priority (smallest numerical value) is processed first.

        - **heapq.heapify**: This function converts the list of jobs into a heap, which is a binary tree structure where the smallest element is always at the root.
        - **heapq.heappop**: This function removes and returns the job with the highest priority from the heap.

        By using this approach, the scheduler can efficiently manage and execute jobs based on their priority, rather than their arrival time.

        Next, we'll explore how a **Round-Robin Scheduler** handles job scheduling.

        """
    )
    return


@app.cell
def _(jobs, time):
    def _round_robin_scheduler(jobs, time_slice):
        print('Round-Robin Scheduling')
        queue = jobs[:]
        current_time = 0
        while queue:
            job = queue.pop(0)
            exec_time = min(time_slice, job[2])
            print(f'Starting {job[1]} at time {current_time} for {exec_time} units')
            time.sleep(1)
            current_time = current_time + exec_time
            if job[2] > time_slice:
                queue.append((job[0], job[1], job[2] - time_slice, job[3]))
            else:
                print(f'Completed {job[1]} at time {current_time}')
    print('\n=== Round-Robin ===')
    _round_robin_scheduler(jobs[:], time_slice=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Explanation of Round-Robin Scheduling

        The **Round-Robin Scheduler** is a preemptive scheduling strategy where each job is given a fixed time slice to execute. If a job doesn't complete within its allotted time, it's put back into the queue to wait for the next round.

        - **Time Slice**: A fixed amount of time (in this example, 2 units) during which each job is allowed to run.
        - **Requeueing**: If a job's execution time exceeds the time slice, the remaining time is requeued, allowing other jobs to execute in the meantime.

        Round-Robin scheduling is commonly used in time-sharing systems where fairness and response time are critical. It ensures that all jobs get a chance to execute within a reasonable timeframe.

        ### Conclusion

        By comparing these three scheduling strategies, you can see how different approaches impact the order and efficiency of job execution:

        - **FCFS** is simple but can lead to long wait times for short jobs if a long job arrives first.
        - **Priority-Based Scheduling** prioritizes jobs based on importance, reducing wait times for critical tasks but potentially delaying lower-priority jobs.
        - **Round-Robin Scheduling** provides a fair allocation of CPU time across all jobs, making it ideal for environments where responsiveness is key.

        ### Questions to Consider

        - How does job priority affect the overall system efficiency?
        - What are the trade-offs between FCFS, Priority-Based, and Round-Robin scheduling?
        - How might different scheduling strategies impact the completion time of individual jobs?

        Experiment with the code by adding more jobs with varying execution times and priorities, and observe how the scheduling order changes. This hands-on approach will give you a deeper understanding of how job schedulers work in HPC environments.

        """
    )
    return


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

        """
    )
    return


@app.cell
def _(heapq, time):
    jobs_1 = [(1, 'Job_1', 4, 2), (3, 'Job_2', 2, 1), (2, 'Job_3', 3, 3), (1, 'Job_4', 5, 2)]

    def _fcfs_scheduler(jobs):
        print('FCFS Scheduling')
        current_time = 0
        for job in jobs:
            print(f'Starting {job[1]} at time {current_time}')
            time.sleep(job[2])
            current_time = current_time + job[2]
            print(f'Completed {job[1]} at time {current_time}')

    def _priority_scheduler(jobs):
        print('Priority-based Scheduling')
        heapq.heapify(jobs)
        current_time = 0
        while jobs:
            job = heapq.heappop(jobs)
            print(f'Starting {job[1]} at time {current_time}')
            time.sleep(job[2])
            current_time = current_time + job[2]
            print(f'Completed {job[1]} at time {current_time}')

    def _round_robin_scheduler(jobs, time_slice):
        print('Round-Robin Scheduling')
        queue = jobs[:]
        current_time = 0
        while queue:
            job = queue.pop(0)
            exec_time = min(time_slice, job[2])
            print(f'Starting {job[1]} at time {current_time} for {exec_time} units')
            time.sleep(exec_time)
            current_time = current_time + exec_time
            if job[2] > time_slice:
                queue.append((job[0], job[1], job[2] - time_slice, job[3]))
            else:
                print(f'Completed {job[1]} at time {current_time}')
    print('=== FCFS ===')
    _fcfs_scheduler(jobs_1)
    print('\n=== Priority-Based ===')
    _priority_scheduler(jobs_1)
    print('\n=== Round-Robin ===')
    _round_robin_scheduler(jobs_1, time_slice=2)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

