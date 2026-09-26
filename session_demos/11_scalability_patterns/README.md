# M3.S1 scalability and parallel-pattern demos

These examples support **M3.S1 — Performance Models, Scalability & Parallel Patterns**.

## Core real cluster mini-lab

`m3s1_cpu_lab.sbatch` requests one 16-core CPU allocation and runs all core experiments sequentially:

1. OpenMP map / strong scaling
2. OpenMP map / weak scaling
3. small workload / thread overhead
4. OpenMP reduction
5. task farm: static vs dynamic scheduling
6. parallel scan
7. OpenMP stencil
8. divide-and-conquer / parallel search
9. MPI reduction with `MPI_Wtime()` and maximum rank time
10. MPI stencil with halo-exchange and compute/communication timing

The batch job uses the validated SciTech `foss/2023b` stack.

## Optional GPU measurement

`m3s1_gpu_lab.sbatch` runs a map kernel and reports both:

- CUDA **kernel time** using CUDA events
- **end-to-end time** including allocation and data movement

This reinforces that measurement boundaries matter.

## Files

- `m3s1_openmp_patterns.c`
- `m3s1_mpi_reduce.c`
- `m3s1_mpi_stencil.c`
- `m3s1_cuda_map.cu`
- `m3s1_cpu_lab.sbatch`
- `m3s1_gpu_lab.sbatch`
