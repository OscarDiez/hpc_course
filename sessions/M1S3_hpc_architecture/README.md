# M1.S3 - Architecture of HPC systems

Interactive notebook for Module 1, Session 3.

The notebook reinforces the architecture vocabulary through short guided exercises and live cluster inspection:

- system -> rack -> node -> processor -> core hierarchy;
- CPU topology;
- shared memory and NUMA;
- distributed memory;
- interconnect latency and bandwidth;
- storage hierarchy;
- CPU/GPU heterogeneous nodes;
- Flynn's taxonomy;
- system software stack;
- SciTech cluster architecture inspection;
- architecture balance;
- JUPITER architecture detective;
- workload-to-architecture matching.

Teaching pattern:

```text
PREDICT -> RUN -> OBSERVE -> EXPLAIN
```

## Open in JupyterHub

Use **File -> Open from URL...** and paste:

```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M1S3_hpc_architecture/M1S3_hpc_architecture.ipynb
```

The notebook performs architecture inspection only. It does not submit compute-heavy Slurm jobs.
