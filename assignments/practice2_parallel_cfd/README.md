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

## SciTech CPU/MPI environment

For MPI and hybrid jobs the scripts start from a clean module environment and load the tested SciTech CPU-HPC stack:

```bash
module purge
module load foss/2023b
```

This avoids mixing the EESSI compatibility compiler with the system OpenMPI installation. The MPI launches also use the tested OpenMPI TCP/vader path to avoid confusing UCX warnings.

For the small MPI demo use:

```bash
bash run_mpi_demo.sh
```

## Accelerator status

GPU allocation itself is available on SciTech and `nvidia-smi` sees the RTX 6000 Ada GPU.

At the latest course validation, however:

- `nvcc` was not available in the active module environment, so CUDA C/C++ compilation cannot yet be required on SciTech;
- `nvc`/NVIDIA HPC SDK was not available;
- GCC accepted `-fopenacc`, but NVIDIA OpenACC offload was not supported by the installed GCC runtime.

Therefore CUDA/OpenACC GPU execution remains conditional on the cluster administrator providing/activating the accelerator toolchain. The source-code TODOs may still be used as implementation exercises, but GPU execution must not be treated as mandatory until the toolchain is confirmed.

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

CUDA/GPU (only when the CUDA toolchain is available):

```bash
JOB_GPU=$(sbatch --parsable jobs/p2_gpu.sbatch | cut -d';' -f1)
```

Monitor with:

```bash
squeue -u $USER
```

## Evidence

After the Slurm jobs finish, run the evidence generator explicitly through Bash:

```bash
bash make_evidence.sh "$JOB_CPU" "$JOB_MPI" "$JOB_HYBRID" "$JOB_GPU"
cat p2_evidence.txt
```

Using `bash` avoids relying on the executable permission bit of files created through the repository interface.

Keep all `.out` files until the assessment has been graded.
