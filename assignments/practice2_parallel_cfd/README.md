# Practice 2 — Parallelize a Mini-CFD Workload

This practice uses one simple 2-D stencil workload to compare the parallel models covered in Module 2:

**serial → OpenMP → MPI → MPI + OpenMP → GPU → compare**

The numerical update is:

```text
new(i,j) = 0.25 * (north + south + west + east)
```

The same deterministic problem is used in every implementation so results can be compared.

## Quick start

From the SciTech login node:

```bash
cd ~/hpc_course
git pull
cd assignments/practice2_parallel_cfd
```

Compile the CPU versions:

```bash
make cpu
```

Run the small deterministic serial demo:

```bash
./bin/stencil_serial --demo
```

For the demo (`21 x 21`, 4 steps, one central hot point of 100), a correct implementation prints:

```text
CENTER_VALUE=14.062500
CHECKSUM=100.000000
NONZERO=25
```

## Submit the experiments

CPU/OpenMP:

```bash
JOB_CPU=$(sbatch --parsable jobs/p2_cpu.sbatch | cut -d';' -f1)
echo $JOB_CPU
```

MPI (4 ranks, 2 nodes):

```bash
JOB_MPI=$(sbatch --parsable jobs/p2_mpi.sbatch | cut -d';' -f1)
echo $JOB_MPI
```

Hybrid MPI + OpenMP:

```bash
JOB_HYBRID=$(sbatch --parsable jobs/p2_hybrid.sbatch | cut -d';' -f1)
echo $JOB_HYBRID
```

GPU:

```bash
JOB_GPU=$(sbatch --parsable jobs/p2_gpu.sbatch | cut -d';' -f1)
echo $JOB_GPU
```

Monitor your jobs with:

```bash
squeue -u $USER
```

The GPU allocation works on SciTech, but the CUDA/OpenACC compiler toolchain may require the accelerated EESSI module tree. The GPU job detects this and reports a clear status if `nvcc` is not yet available.

## Evidence

After the jobs finish:

```bash
./make_evidence.sh "$JOB_CPU" "$JOB_MPI" "$JOB_HYBRID" "$JOB_GPU"
cat p2_evidence.txt
```

Keep all `.out` files until the assessment has been graded.
