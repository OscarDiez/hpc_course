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

## D. CUDA compiler availability on the SciTech cluster

`nvidia-smi` confirms that a GPU and NVIDIA driver are available, but it does **not** mean that the CUDA compiler (`nvcc`) is available in the shell.

As tested on 5 September 2026, the GPU allocation works, but `nvcc` is not in the default PATH and the normal module tree does not expose CUDA:

```bash
which nvcc
module avail 2>&1 | grep -Ei 'cuda|nvhpc|nvidia'
```

SciTech has indicated that CUDA, cuDNN, NCCL and CUDA-Samples are available through an accelerated EESSI module tree that is not enabled by default. The exact site command still needs to be confirmed/enabled by SciTech before students are asked to compile CUDA on the cluster.

Instructor diagnostics:

```bash
module spider CUDA-Samples
module spider CUDA
```

If SciTech exposes the accelerated module tree, load the CUDA/CUDA-Samples module they provide and verify:

```bash
which nvcc
nvcc --version
```

Until that is enabled, use the cluster for the **Slurm + GPU allocation + `nvidia-smi`** demonstration, and run the CUDA code in the course notebook/Colab environment instead.

## E. Demo 1 — Vector addition (once `nvcc` is enabled)

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

## F. Demo 2 — CPU vs GPU benchmark (once `nvcc` is enabled)

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
