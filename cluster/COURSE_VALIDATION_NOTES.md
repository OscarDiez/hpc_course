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

Confirmed working:
- Slurm `gpu` partition.
- GPU allocation.
- NVIDIA RTX 6000 Ada Generation visible with `nvidia-smi`.
- Shared CUDA Apptainer image:
  `/data/software/containers/hpc-course-cuda.sif`.
- CUDA 12.8 path through the shared image has already been validated in Practice 2.

Issues / limitations already observed:
- native `nvcc` was not exposed in the default GPU job environment;
- NVIDIA HPC SDK `nvc` was not exposed;
- CuPy was not installed in the default Python environment;
- OpenACC is currently treated as a code/concept exercise unless an accelerator-capable compiler environment is provided.

Questions for Francisco before relying on native GPU tooling:
- Is native CUDA expected to be available through a supported module or system path?
- Is the shared Apptainer CUDA image the recommended student route?
- Is NVIDIA HPC SDK planned/supported for OpenACC exercises?
- Are there any JupyterHub-to-Slurm environment variables that should be stripped for GPU jobs, similar to the CPU/MPI case?
