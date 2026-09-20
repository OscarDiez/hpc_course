# M1.S1 - Fundamentals of HPC

## From a serial problem to a supercomputer

Interactive 2026 notebook for Module 1, Session 1.

The notebook follows the live session objectives:

- identify when a workload becomes an HPC problem;
- distinguish time-to-solution from throughput;
- distinguish concurrency, data parallelism and task parallelism;
- establish a serial baseline;
- reason about scaling with Amdahl's Law;
- identify compute, memory, network, storage and software bottlenecks;
- inspect the SciTech cluster safely without launching heavy work.

Teaching pattern:

```text
PREDICT -> RUN -> OBSERVE -> EXPLAIN
```

## Open in JupyterHub

Use **File -> Open from URL...** and paste:

```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M1S1_fundamentals/M1S1_fundamentals.ipynb
```

## Execution model

Most examples run directly in the Jupyter kernel and are intentionally small.

The final section only inspects the SciTech environment with commands such as `hostname`, `lscpu`, `sinfo` and `squeue`. It does not submit a heavy Slurm job because resource allocation and job submission are introduced later in Module 1.

Legacy M1.S1 material remains under `chapters_examples/`.
