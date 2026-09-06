# Practice 2 — One Problem, Four Parallel Models

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

## Build targets

```bash
make serial
make openmp
make mpi
make openacc
make cuda
make hybrid
```

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

The parallel implementations should reproduce these numerical values.

## SciTech environment

CPU/OpenMP and MPI use the standard SciTech CPU environment. The supplied Slurm scripts load `foss/2023b` if needed.

CUDA requires `nvcc`. GPU allocation on SciTech is available, but the exact accelerated EESSI/CUDA activation command still needs to be confirmed by the cluster administrator. The GPU job detects a missing `nvcc` and reports `GPU_TOOLCHAIN_STATUS=NVCC_NOT_AVAILABLE` rather than failing silently.

OpenACC compilation is supported by the Makefile through `ACC_CC` and `ACCFLAGS`. By default it uses GCC with `-fopenacc`; actual NVIDIA GPU offload depends on the OpenACC-capable compiler environment provided on SciTech. Until that environment is confirmed, OpenACC can be used for the implementation/compiler exercise but GPU-offload results should not be assumed.

## Slurm experiments

CPU/OpenMP:

```bash
JOB_CPU=$(sbatch --parsable jobs/p2_cpu.sbatch | cut -d';' -f1)
```

MPI (4 ranks, 2 nodes):

```bash
JOB_MPI=$(sbatch --parsable jobs/p2_mpi.sbatch | cut -d';' -f1)
```

Hybrid MPI + OpenMP:

```bash
JOB_HYBRID=$(sbatch --parsable jobs/p2_hybrid.sbatch | cut -d';' -f1)
```

CUDA/GPU:

```bash
JOB_GPU=$(sbatch --parsable jobs/p2_gpu.sbatch | cut -d';' -f1)
```

Monitor with:

```bash
squeue -u $USER
```

## Evidence

After the four Slurm jobs finish:

```bash
./make_evidence.sh "$JOB_CPU" "$JOB_MPI" "$JOB_HYBRID" "$JOB_GPU"
cat p2_evidence.txt
```

Keep all `.out` files until the assessment has been graded.
