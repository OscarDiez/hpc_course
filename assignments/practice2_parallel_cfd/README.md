# Practice 2 — One Problem, Five Parallel Models

This practice uses one simple 2-D stencil workload to compare the parallel models covered in Module 2:

**serial → OpenMP → MPI → OpenACC → CUDA → hybrid → compare**

The numerical update is:

```text
new(i,j) = 0.25 * (north + south + west + east)
```

The same deterministic problem is used in every implementation so results can be compared.

## Student files

```text
src/stencil_serial.c
src/stencil_openmp_student.c
src/stencil_mpi_student.c
src/stencil_openacc_student.c
src/stencil_cuda_student.cu
src/stencil_hybrid.c
```

The `_student` files contain small TODO sections. Complete only those sections requested in the Blackboard assignment.

## Reference result

Start with:

```bash
make serial
./bin/stencil_serial --demo
```

For the deterministic demo (`21 x 21`, 4 steps, one central hot point of 100), a correct implementation gives:

```text
CENTER_VALUE=14.062500
CHECKSUM=100.000000
NONZERO=25
```

All correct implementations should reproduce these numerical values.

## Validated SciTech CPU workflow

The supplied Slurm scripts use a clean SciTech `foss/2023b` environment so that student jobs do not depend on modules inherited from the login shell.

The validated CPU jobs are:

```bash
JOB_CPU=$(sbatch --parsable jobs/p2_cpu.sbatch | cut -d';' -f1)
JOB_MPI=$(sbatch --parsable jobs/p2_mpi.sbatch | cut -d';' -f1)
JOB_HYBRID=$(sbatch --parsable jobs/p2_hybrid.sbatch | cut -d';' -f1)
```

Current SciTech compute nodes do not expose the `srun` client command needed by OpenMPI's normal Slurm launcher. For a robust course workflow, the supplied MPI and hybrid jobs therefore run inside one allocated compute node and explicitly exclude OpenMPI's Slurm launcher.

MPI still uses separate processes, separate address spaces and real halo-message exchange, so the MPI programming model is demonstrated correctly.

The validated reference configurations are:

```text
MPI:      4 ranks
Hybrid:   2 MPI ranks x 4 OpenMP threads
```

## CUDA accelerator route

GPU allocation works on SciTech.

The supplied GPU job automatically tries:

1. native `nvcc`, if available;
2. otherwise the shared course Apptainer image:

```text
/data/software/containers/hpc-course-cuda.sif
```

Students therefore use one command:

```bash
JOB_GPU=$(sbatch --parsable jobs/p2_gpu.sbatch | cut -d';' -f1)
```

The output reports the route used:

```text
GPU_TOOLCHAIN_ROUTE=NATIVE_NVCC
```

or:

```text
GPU_TOOLCHAIN_ROUTE=APPTAINER_SHARED
```

The shared Apptainer CUDA route has been validated on the SciTech RTX 6000 Ada GPU with CUDA 12.8.

Students must not download their own CUDA container image.

## OpenACC status

At the latest release validation, NVIDIA HPC SDK (`nvc`, `nvc++`, `nvfortran`) was not yet available and the installed GCC OpenACC runtime could not execute the accelerator program correctly.

For the published assignment, OpenACC is therefore a **code and concept task only** unless the instructor announces that an OpenACC-capable environment has become available.

Students should complete the requested OpenACC directives and understand the role of the data region and parallel loop. They should not attempt to install their own compiler or container.

## Monitor jobs

```bash
squeue -u $USER
```

Inspect output after completion, for example:

```bash
cat p2_cpu_${JOB_CPU}.out
cat p2_mpi_${JOB_MPI}.out
cat p2_hybrid_${JOB_HYBRID}.out
cat p2_gpu_${JOB_GPU}.out
```

## Evidence

If CUDA ran as a SciTech Slurm job:

```bash
bash make_evidence.sh "$JOB_CPU" "$JOB_MPI" "$JOB_HYBRID" "$JOB_GPU"
```

If a non-Slurm accelerator fallback is announced instead:

```bash
bash make_evidence.sh "$JOB_CPU" "$JOB_MPI" "$JOB_HYBRID" NO_SLURM_GPU
```

Then inspect:

```bash
cat p2_evidence.txt
```

Keep all source files and `.out` files until the assessment has been graded.
