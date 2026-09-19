# M2.S2 — Shared-memory computing with OpenMP

Interactive notebook for **M2.S2 — Shared-memory computing with OpenMP**.

## Open in IE/SciTech JupyterHub

1. Open: https://jupyter.iesci.tech:8085/
2. Choose **File → Open from URL...**
3. Paste:

```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M2S2_openmp/M2S2_openmp.ipynb
```

4. Use the **Python 3** kernel.
5. Save a personal copy if you want to preserve changes.

## Classroom flow

**PREDICT → RUN → OBSERVE → EXPLAIN**

Activities:
1. environment/compiler check
2. fork–join and thread IDs
3. `parallel for` worksharing
4. shared/private data
5. race condition
6. reduction and timing
7. static vs dynamic scheduling
8. Slurm cores ↔ OpenMP threads
9. final safe-parallelization challenge

## Notes

- The notebook does not install GCC or packages.
- C examples compile with `gcc -fopenmp`.
- Slurm-specific steps are clearly marked and fail gracefully outside the cluster.
- Legacy OpenMP notebooks remain under `chapters_examples/` and `assignments/` for reference.