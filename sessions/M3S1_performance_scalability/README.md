# M3.S1 — Performance Models, Scalability & Parallel Patterns

Interactive notebook for the 2026 HPC course.

## Classroom method

**PREDICT → RUN → OBSERVE → EXPLAIN**

## Core real cluster workflow

The notebook submits one short CPU/MPI Slurm job using the demos in:

`session_demos/11_scalability_patterns/`

It measures:

1. OpenMP map / strong scaling
2. OpenMP map / weak scaling
3. small workload / too many threads
4. OpenMP reduction
5. task farm: static vs dynamic scheduling
6. parallel scan
7. OpenMP stencil
8. parallel search / divide work
9. MPI reduction with maximum rank time
10. MPI stencil with halo-exchange timing

## Parallel patterns

Every pattern taught has a concrete example:

- Map / data parallel
- Reduce
- Scan / prefix
- Stencil
- Domain decomposition
- Task farm / manager-worker
- Pipeline / dataflow
- Divide and conquer / parallel search

## OpenMP, MPI and GPU measurement

The notebook shows:

- `omp_get_wtime()`
- `MPI_Wtime()` plus maximum elapsed rank time
- CUDA events for kernel timing
- end-to-end GPU timing including data movement

The GPU example is optional and runs through a short GPU Slurm job.

## Dependencies

The notebook does not require matplotlib, NumPy or pandas.

## Open from URL

https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M3S1_performance_scalability/M3S1_performance_scalability.ipynb
