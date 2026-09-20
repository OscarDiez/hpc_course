# M2.S3 — Distributed-memory computing with MPI

Interactive notebook for **M2.S3 — Distributed-memory computing with MPI**.

## Open in IE/SciTech JupyterHub

1. Open: https://jupyter.iesci.tech:8085/
2. Choose **File → Open from URL...**
3. Paste:

```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M2S3_mpi/M2S3_mpi.ipynb
```

4. Use the **Python 3** kernel.
5. Save a personal copy if you want to preserve changes.

## Classroom flow

**PREDICT → RUN → OBSERVE → EXPLAIN**

Activities:
1. MPI/Slurm environment check
2. rank, size and processor identity
3. separate process memory
4. point-to-point send/receive/tags
5. deadlock reasoning and safe message ordering
6. broadcast/scatter/gather/reduce
7. communication-cost ping-pong
8. Slurm multi-node mapping
9. distributed image-processing challenge

## Notes

- The notebook does not install MPI or packages.
- MPI C examples compile with `mpicc`.
- Local `mpirun`/`mpiexec` execution is used only when the environment permits it.
- The multi-node Slurm script is generated but not submitted automatically.
- Legacy MPI notebooks remain under `chapters_examples/` and `assignments/` for reference.

## SciTech validation note - 20 September 2026

The current robust classroom workflow is:

- small MPI examples directly from Jupyter;
- a real 4-rank MPI batch job on one CPU node;
- a separate 2-node Slurm placement demonstration.

A true 2-node MPI executable was not made the default because validation found an inconsistent MPI runtime environment between the Jupyter/batch/compute-node contexts. The cluster administrator should confirm the supported multi-node MPI stack before it is required from students.
