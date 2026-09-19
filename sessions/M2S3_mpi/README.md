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