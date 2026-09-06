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

## SciTech CPU/MPI environment

For MPI and hybrid jobs the scripts start from a clean module environment and load the tested SciTech CPU-HPC stack:

```bash
module purge
module load foss/2023b
```

This avoids mixing the EESSI compatibility compiler with the system OpenMPI installation.

SciTech compute nodes currently do not expose the `srun`/`scontrol` client commands required by OpenMPI's normal Slurm launcher. For a robust student workflow, the supplied MPI and hybrid jobs therefore run on **one allocated compute node** and explicitly exclude OpenMPI's Slurm process launcher. MPI still uses separate processes with separate address spaces and real message passing/halo exchange, so the exercise demonstrates the MPI programming model correctly without depending on unfinished site launcher integration.

The jobs also use the tested OpenMPI TCP/vader communication path to avoid confusing UCX warnings.

For the small deterministic MPI demo use:

```bash
bash run_mpi_demo.sh
```

## Accelerator route

GPU allocation is available on SciTech and `nvidia-smi` sees the NVIDIA RTX 6000 GPU.

The supplied GPU batch script automatically chooses the first working CUDA route:

1. **native `nvcc`**, if the cluster accelerator modules provide it;
2. otherwise the centrally shared Apptainer image:

```text
/data/software/containers/hpc-course-cuda.sif
```

The student command is therefore the same in either case:

```bash
JOB_GPU=$(sbatch --parsable jobs/p2_gpu.sbatch | cut -d';' -f1)
```

The output reports either:

```text
GPU_TOOLCHAIN_ROUTE=NATIVE_NVCC
```

or:

```text
GPU_TOOLCHAIN_ROUTE=APPTAINER_SHARED
```

Students must **not** download their own multi-GB CUDA container image.

If neither native CUDA nor the shared image is available, the GPU job reports the infrastructure limitation explicitly. A course fallback environment may then be used as described in Blackboard.

### OpenACC

At the latest validation, NVIDIA HPC SDK (`nvc`, `nvc++`, `nvfortran`) was not yet available. Students should complete the OpenACC directives in the source code. GPU execution is required only if an OpenACC-capable accelerator environment is announced before the practice.

## Slurm experiments

CPU/OpenMP:

```bash
JOB_CPU=$(sbatch --parsable jobs/p2_cpu.sbatch | cut -d';' -f1)
```

MPI (4 MPI ranks on one allocated compute node):

```bash
JOB_MPI=$(sbatch --parsable jobs/p2_mpi.sbatch | cut -d';' -f1)
```

Hybrid MPI + OpenMP (2 MPI ranks × 4 OpenMP threads on one allocated compute node):

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

If CUDA ran as a SciTech Slurm job:

```bash
bash make_evidence.sh "$JOB_CPU" "$JOB_MPI" "$JOB_HYBRID" "$JOB_GPU"
```

If the CUDA part had to use a non-Slurm course fallback:

```bash
bash make_evidence.sh "$JOB_CPU" "$JOB_MPI" "$JOB_HYBRID" NO_SLURM_GPU
```

Then inspect:

```bash
cat p2_evidence.txt
```

Using `bash` avoids relying on the executable permission bit of files created through the repository interface.

Keep all `.out` files until the assessment has been graded.
