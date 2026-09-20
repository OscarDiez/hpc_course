# M1.S4 - Resource management and performance metrics

Interactive notebook for Module 1, Session 4.

The notebook turns the session into a guided SciTech exercise:

- inspect Slurm partitions, nodes and the current Jupyter allocation;
- understand nodes, tasks, CPUs per task, memory, GPUs and wall time;
- create and submit a real batch job;
- inspect PENDING/RUNNING states and output;
- use `sacct` for completed-job accounting;
- reason about scheduling and backfilling;
- compare sensible and wasteful resource requests;
- calculate speedup and parallel efficiency;
- demonstrate that allocating more CPUs does not parallelize serial code;
- explore Amdahl's Law;
- distinguish peak and sustained performance;
- match HPL, HPCG, STREAM, MLPerf and Green500 to the question they answer.

Teaching pattern:

```text
PREDICT -> RUN -> OBSERVE -> EXPLAIN
```

## Open in JupyterHub

Use **File -> Open from URL...** and paste:

```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M1S4_resource_management/M1S4_resource_management.ipynb
```

The real Slurm jobs are intentionally small. The notebook submits CPU-only jobs to the `cpu` partition and cleans inherited Jupyter memory-allocation variables before calling `sbatch`.
