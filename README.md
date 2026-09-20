# HPC 

Support and exercise files for **HPC**, including the books and the course designed to help you learn and practice high-performance computing concepts. Exercises are provided to be run in an HPC cluster, and some can also be executed in Google Colab.

## Structure of the Repository

- **Assignments**: Practical assignments for each module of the course.
- **Chapter Examples**: Example notebooks to help you practice the concepts covered in the lessons.
- **Cluster guide**: Reusable SciTech HPC cluster access and Slurm instructions.
- **Session demos**: Small reproducible examples used during live sessions.

---

## SciTech HPC cluster

[Reusable cluster access guide](cluster/README.md)

[Session 9 GPU demo](session_demos/09_gpu/README.md)

---

## Loading Notebooks from GitHub in JupyterHub

You can load the following notebooks directly from GitHub into JupyterHub by using the "File -> Open from URL" option in Jupyter.

### Assignments

[M1.P1. Intro to Slurm](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/assignments/M1.P1.IntroSlurm.ipynb)

[M1.P2. Simple Parallel Codes](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/assignments/M1.P2.SimpleParallelCodes.ipynb)

[M2.P1. OpenMP](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/assignments/M2.P1.OpenMP.ipynb)

[M2.P2. MPI](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/assignments/M2.P2.MPI.ipynb)

[M3.P1. Performance Tuning](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/assignments/M3.P1.Performance_tuning.ipynb)


---

### Chapter Examples (Lessons)


[1.M1.S1. Evolution and Fundamentals of HPC (1.1)](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/chapters_examples/1.M1.S1.Evolution%20and%20Fundamentals%20of%20HPC%20(1.1).ipynb)

[2.M1.S2. Architectural Overview of HPC Systems (1.2)](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/chapters_examples/2.M1.S2.Architectural%20Overview%20of%20HPC%20Systems%20(1.2).ipynb)

[3.M1.S3. Resource Management & Performance Metrics in Parallel](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/chapters_examples/3.M1.S3.Resource%20Management%20%26%20Performance%20Metrics%20in%20Parallel.ipynb)

[4.M1.S4. Cloud-based HPC and Virtualization Containers in HPC](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/chapters_examples/4.M1.S4.Cloud-based%20HPC%20and%20Virtualization_Containers%20in%20HPC.ipynb)

[5.M1.S5. HPC in Health & Neurosciences (1.5)](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/chapters_examples/5.M1.S5.HPC%20in%20Health%20%26%20Neurosciences%20(1.5).ipynb)

[7.M2.S2. OpenMP (2.2)](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/chapters_examples/7.M2.S2.OpenMP%20(2.2).ipynb)

[8.M2.S3. Deep Dive MPI (2.3)](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/chapters_examples/8.M2.S3.Deep%20Dive%20MPI%20(2.3).ipynb)

[9.M2.S4. GPU Computing, OpenACC, CUDA Basics (2.4)](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/chapters_examples/9.M2.S4.GPU%20Computing,%20OpenAcc_CUDA%20basics%20(2.4).ipynb)

---

## How to Use

1. **HPC Cluster**: These notebooks are meant to be executed on an HPC cluster, but some can also run on platforms like Google Colab.
2. **Direct URL Loading**: Use the provided raw GitHub links to load the notebooks directly into Jupyter or JupyterHub using the "File -> Open from URL" option.

---

### Contributions

Feel free to contribute to the course materials by creating pull requests, adding additional exercises, or reporting issues.


---

## 2026 Interactive Session Notebooks

Current teaching notebooks are organized by **Module → Session** using the canonical `MxSy` identifier.

### Module 2

- **M2.S1 — Parallel Thinking and Decomposition**
  - [Session folder](sessions/M2S1_parallel_thinking/)
  - [Notebook](sessions/M2S1_parallel_thinking/M2S1_parallel_thinking.ipynb)
  - Raw URL for JupyterLab **File → Open from URL...**:
    `https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M2S1_parallel_thinking/M2S1_parallel_thinking.ipynb`

- **M2.S2 — Shared-memory computing with OpenMP**
  - [Session folder](sessions/M2S2_openmp/)
  - [Notebook](sessions/M2S2_openmp/M2S2_openmp.ipynb)
  - Raw URL for JupyterLab **File → Open from URL...**:
    `https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M2S2_openmp/M2S2_openmp.ipynb`

- **M2.S3 — Distributed-memory computing with MPI**
  - [Session folder](sessions/M2S3_mpi/)
  - [Notebook](sessions/M2S3_mpi/M2S3_mpi.ipynb)
  - Raw URL for JupyterLab **File → Open from URL...**:
    `https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M2S3_mpi/M2S3_mpi.ipynb`

- **M2.S4 — Introduction to GPU and Accelerator Computing**
  - [Session folder](sessions/M2S4_gpu_accelerators/)
  - [Notebook](sessions/M2S4_gpu_accelerators/M2S4_gpu_accelerators.ipynb)
  - Raw URL for JupyterLab **File → Open from URL...**:
    `https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M2S4_gpu_accelerators/M2S4_gpu_accelerators.ipynb`

- **M2.S5 - Asynchronous Guided Lab: One Parallel Application**
  - [Session folder](sessions/M2S5_async_parallel_app/)
  - [Notebook](sessions/M2S5_async_parallel_app/M2S5_async_parallel_app.ipynb)
  - Raw URL for JupyterLab **File -> Open from URL...**:
    `https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M2S5_async_parallel_app/M2S5_async_parallel_app.ipynb`

> Legacy notebooks remain in `chapters_examples/` and `x_archive/`. New 2026 notebooks use `sessions/MxSy_topic/`.