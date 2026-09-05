# Session 9 — GPU Computing Demo

This demo connects the CUDA concepts from the lecture with a real GPU on the SciTech HPC cluster.

## Learning goals

After the demo, you should be able to explain:

- how to request a GPU through Slurm;
- how to identify the GPU assigned to your job;
- how a CUDA vector-add kernel maps work to blocks and threads;
- why GPU kernel time and total application time are not the same;
- why data movement and problem size affect speedup.

## A. Access the cluster

See the reusable guide:

`cluster/README.md`

From outside campus:

```bash
ssh -J <username>@ssh.iesci.tech <username>@10.205.20.10
```

Then check the GPU partition:

```bash
sinfo -p gpu
```

## B. Request one GPU

For the short classroom demo:

```bash
srun -p gpu --gpus=1 --pty bash
```

Check the allocated compute node and GPU:

```bash
hostname
nvidia-smi
```

When you finish, type `exit` to release the allocation.

## C. Get the examples

On the login node (`rust`), clone the course repository once:

```bash
git clone https://github.com/OscarDiez/hpc_course.git
cd hpc_course/session_demos/09_gpu
```

If you already cloned it:

```bash
cd ~/hpc_course
git pull
cd session_demos/09_gpu
```

## D. Enable CUDA

The GPU driver alone is not enough to compile CUDA code. `nvidia-smi` can show a CUDA compatibility version even when `nvcc` is not in your PATH.

SciTech indicated that CUDA is provided through the accelerated EESSI module tree. Before class, the instructor will confirm the exact module command. To discover available modules:

```bash
module avail 2>&1 | grep -Ei 'cuda|nvhpc|nvidia'
```

After loading the CUDA module, verify:

```bash
which nvcc
nvcc --version
```

If `nvcc` is still unavailable, do not troubleshoot for a long time during class: run the notebook/Colab version and use the cluster only for the Slurm + `nvidia-smi` demonstration.

## E. Demo 1 — Vector addition

Compile:

```bash
nvcc vector_add.cu -O2 -o vector_add
```

Run:

```bash
./vector_add
```

The program adds two vectors with 1024 elements using:

- 256 threads per block;
- 4 blocks;
- one useful CUDA thread per vector element.

Think before running: **Why are 4 blocks enough?**

## F. Demo 2 — CPU vs GPU benchmark

Compile:

```bash
nvcc vector_benchmark.cu -O3 -o vector_benchmark
```

Run:

```bash
./vector_benchmark
```

The program prints three timings for several vector sizes:

- CPU time;
- GPU kernel-only time;
- GPU total time, including CPU→GPU and GPU→CPU transfers.

### Questions

1. For which size does the CPU win?
2. When does the GPU become competitive?
3. Why is `GPU kernel` faster than `GPU total`?
4. Which GPU number should be compared with the CPU if the application must copy input and output every time?
5. What changes when data can stay on the GPU for several operations?

Do not expect exactly the same timings on every run or every GPU. The important result is the **pattern**, not a particular number.

## Good cluster citizenship

GPU resources are shared and limited. For a class with many students:

- keep interactive tests short;
- release the allocation with `exit` when finished;
- do not all request GPUs simultaneously unless instructed;
- use the notebook/Colab version for individual practice when the GPU queue is busy.
