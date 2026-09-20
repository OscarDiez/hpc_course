# HPC Course 2026 - SciTech validation notes

Last updated: 20 September 2026

This file records infrastructure findings discovered while validating the Module 2 teaching notebooks from JupyterHub.

## MPI / multi-node CPU

Confirmed working:
- Slurm `cpu` partition.
- Two-node allocations using `haskell` and `julia`.
- Four Slurm tasks split 2 + 2 across the two nodes.
- Local MPI examples from Jupyter.
- One-node MPI execution with separate MPI processes.

Issues observed when attempting true two-node MPI from a Jupyter-submitted batch job:
1. JupyterHub exports multiple `SLURM_MEM_PER_*` variables into submitted jobs. Slurm reports that `SLURM_MEM_PER_CPU`, `SLURM_MEM_PER_GPU`, and `SLURM_MEM_PER_NODE` are mutually exclusive unless they are explicitly removed.
2. The non-interactive batch shell does not expose the `module` command. Attempts to initialise it from `/etc/profile.d/modules.sh` or `/usr/share/Modules/init/bash` did not provide Environment Modules in the tested job.
3. A program compiled with the MPI visible from the Jupyter environment could start on `haskell`, but ranks placed on `julia` failed with:
   `libmpi.so.40: cannot open shared object file: No such file or directory`.
4. The repository's previously validated Practice 2 workflow also documents that the current SciTech setup is not using a normal multi-node OpenMPI/Slurm launcher path and therefore keeps its MPI run on one allocated node.

Questions for Francisco / cluster administration:
- What is the supported MPI stack for student multi-node jobs?
- Should OpenMPI or another MPI implementation be installed identically on `haskell` and `julia`?
- Is there a centrally supported module/Lmod initialisation path for non-interactive Slurm jobs?
- Can JupyterHub avoid exporting conflicting `SLURM_MEM_PER_*` variables into child `sbatch` jobs?
- What launcher should students use for multi-node MPI: `srun --mpi=...`, `mpirun`, or another site-specific wrapper?

## GPU / accelerator

Confirmed working in the current M2.S4 validation:
- Slurm `gpu` partition.
- GPU allocation.
- NVIDIA RTX 6000 Ada Generation visible with `nvidia-smi`.
- Job 19678 received 49140 MiB GPU memory and driver 580.126.09.

Current issues / limitations observed in the Jupyter-submitted GPU batch job:
- native `nvcc` is not exposed;
- NVIDIA HPC SDK `nvc` is not exposed;
- CuPy is not installed in the default Python environment;
- the expected shared image path `/data/software/containers/hpc-course-cuda.sif` did not provide a usable route in the current job;
- the previously documented Practice 2 CUDA container path should therefore be treated as historically validated, not assumed to be currently available from every GPU batch environment.

The M2.S4 notebook now explicitly probes:
- native `nvcc`;
- `hpc-course-cuda/12.8.1` through a login shell;
- `nvhpc/25.7` through a login shell;
- the shared Apptainer image;
- native and module-based `nvc`;
- availability/readability diagnostics for the container and module environment.

Questions for Francisco before relying on native GPU tooling:
- Is native CUDA expected to be available through a supported module or system path?
- Is the shared Apptainer CUDA image the recommended student route?
- Is NVIDIA HPC SDK planned/supported for OpenACC exercises?
- Are there any JupyterHub-to-Slurm environment variables that should be stripped for GPU jobs, similar to the CPU/MPI case?
