# M5.S1 — Hybrid computing patterns (2026–27)

This session corresponds to syllabus **Session 21: Hybrid computing patterns**. It is a compact, student-facing SciTech Jupyter notebook and accompanies the slide sequence.

## Open in SciTech JupyterHub

1. Navigate to https://jupyter.iesci.tech:8085/ and select Python 3.
2. Use **File → Open from URL...** and paste the raw notebook URL for this branch:

`https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s1-hybrid-patterns-2026/sessions/M5S1_hybrid_computing/M5S1_hybrid_computing.ipynb`

3. Save your own copy before changing settings. The notebook writes its C/CUDA sources to a user-owned shared workspace; it does not install system software.

## Student route

Predict → setup → materialize source → submit **one Slurm CPU job** → collect → inspect rank/thread mapping → inspect actual temperature grids → compare checksum, whole-grid correctness and elapsed times → optional GPU job → change one parameter → explain.

The CPU batch uses one Slurm CPU node, two tasks, two CPUs/task, 2 GiB, six-minute limit. It executes C MPI+OpenMP code with halo exchange and numerical checks, comparing 1×1, 2×1 and 2×2 rank/thread configurations for the same small heat problem.

A separate optional single-GPU vector-add lab has **`RUN_GPU=True` by default**. When its cell executes, it submits **one short GPU batch job** using normal `sbatch` (no `--immediate` flag). **Slurm queues the job when GPUs are occupied** and executes it when one is allocated, even if Jupyter is closed. The submitted job ID is saved and reused to prevent duplicate jobs. The GPU collection cell reports **QUEUED**, **RUNNING**, **PASS**, **FAIL**, or **SKIPPED**, and waits up to 20 seconds by default before returning while the Slurm job continues. Rerun the collection cell later to inspect the actual output. The GPU lab compares kernel-only and end-to-end duration and verifies arithmetic; it is **not** a multi-GPU MPI experiment.

## Realism and restrictions

- Build: `M5S1-2026-10-13-v2`.
- MPI: SciTech `OpenMPI/4.1.5-GCC-12.3.0` first, compatible fallback when present. Compiled on allocated node with `mpicc -fopenmp`, launched by `srun`.
- GPU: `nvhpc/25.7` when available and allocated via Slurm. The optional GPU job is **automatically submitted and allowed to wait in Slurm's queue by default**, unless `RUN_GPU=False`. Slurm is responsible for the GPU allocation; the notebook never reserves or preempts busy GPUs. To cancel a pending or running job, use `scancel <job_id>`. Do not set `GPU_NEW_RUN=True` unless deliberately starting an additional experimental run.
- `PASS` requires real outputs and numerical checks. `FAIL`, `PENDING`, `NOT RUN` and `SKIPPED` are distinct.
- Measured time is max elapsed across ranks for the heat update and halo loop; compilation and result writing are excluded. Different CPU allocations mean this is not a fair isolated OpenMP scaling benchmark. Repeat timing runs to understand variability.
- The default workspace must be a **shared filesystem** accessible from allocated compute nodes; change `WORKSPACE` if needed. The notebook never SSHs directly to worker nodes.
- **Compatibility fix:** the October 13 v2 HTML export showed `sbatch: unrecognized option '--immediate=15'` even though `sinfo` indicated GPU capacity. The GPU notebook now uses ordinary `sbatch` and permits queuing.
- We have **not executed this new queueing revision on SciTech**. A live Slurm CPU and optional GPU run must be verified before presenting screenshots or timings as SciTech evidence. Existing notebooks' successful older runs are not validation of this one.

See [SciTech guide](../../cluster/README.md), [earlier MPI notebook](../M2S3_mpi/M2S3_mpi.ipynb), and [earlier GPU notebook](../M2S4_gpu_accelerators/M2S4_gpu_accelerators.ipynb).
