# M2.S5 - Asynchronous Guided Lab

## One application, three parallel models, one real HPC system

This 90-minute self-guided notebook consolidates Module 2 around one simple scientific application: a two-dimensional heat-diffusion stencil.

Core path:

- serial baseline;
- OpenMP shared-memory execution;
- MPI distributed-memory execution with halo exchange;
- comparison of correctness, timing and scaling.

Optional accelerator extension:

- OpenACC on the SciTech GPU through `nvhpc/25.7`.

The notebook follows the classroom pattern:

```text
PREDICT -> RUN -> OBSERVE -> EXPLAIN
```

The core activity aligns with the syllabus description of Practice 2: students run a sequential program, parallelize/execute with OpenMP, execute MPI, compare shared-memory and distributed-memory behavior, retain execution logs and basic timings, and explain the differences.

## Open in JupyterHub

Use **File -> Open from URL...** and paste:

```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M2S5_async_parallel_app/M2S5_async_parallel_app.ipynb
```

## Current SciTech assumptions

Validated Module 2 infrastructure:

- CPU Slurm jobs work;
- OpenMP works;
- real MPI with multiple ranks works reliably on one allocated CPU node;
- GPU allocation works on the RTX 6000 Ada;
- `nvhpc/25.7` provides CUDA and OpenACC tooling;
- multi-node MPI is not required by this notebook.

The GPU section is an extension so the syllabus core remains serial + OpenMP + MPI.


## Running the notebook

Use **Kernel -> Restart Kernel and Run All Cells**. The notebook creates source files and Slurm scripts progressively, so students should not jump directly to later submission cells.

The notebook does not require Matplotlib. Visualizations use small inline SVG/HTML helpers so it works with the standard JupyterHub Python environment.


## Performance-design note

The validated v2 run worked end to end, but it exposed two avoidable sources of CPU overhead: clearing the whole output grid every timestep and creating a new OpenMP parallel region for every timestep. Build v3 removes the redundant clearing and keeps one persistent OpenMP team across the full simulation. This makes the serial/OpenMP/MPI/GPU comparison more representative and leaves memory bandwidth and synchronization as the main scaling effects students should interpret.
