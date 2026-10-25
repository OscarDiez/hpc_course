# M5S5 · Vector search integration lab

Guided C/Slurm lab consolidating Module 5, with a required Apptainer result-validation workflow. The individual cluster assignment, module quiz and group milestone are separate activities.

[Notebook](M5S5_vector_search_integration.ipynb)

Raw URL for JupyterLab **File → Open from URL**:

https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s5-vector-search-2026/sessions/M5S5_vector_search_integration/M5S5_vector_search_integration.ipynb

Save your own copy and run the Python 3 cells in order. The notebook embeds all C, Python, shell and batch files and uses the owned `m5s5_vector_lab` folder under the notebook working directory. No package installation or image download is performed.

## Module 5 practice map

| Earlier session | Actual practice here |
|---|---|
| M5S1: hybrid patterns | MPI shards, local OpenMP work, equal four-CPU layout comparison where site MPI works |
| M5S2: scalable applications | Independent correctness fixture, actual team counts, repeated timings, communication/selection evidence |
| M5S3: portability/reproducibility | Inspect modules/wrapper paths; execute Python result validation inside an existing Apptainer image; binds, clean environment, explicit variable forwarding, image/input/script hashes |
| M5S4: project workshop | Keep a runnable, traceable compute-to-validation workflow and explain limitations for a project handover |

The C search uses deterministic synthetic vectors; it does not evaluate semantic relevance. Native C outputs are checked against serial/independent references. The core container stage replays validation and analysis of these files; it does not claim C execution inside the Python image.

## Run

Use the [SciTech guide](../../cluster/README.md). Work under your shared home directory, not a compute node's `/tmp`. The supplied `site_env.sh` selects the configuration used in instructor SciTech job 24943: system tools first, `OMPI_CC=/usr/bin/gcc`, and shared image `/data/software/containers/pytorch-2.8.0-cuda12.8.sif`. Each batch job sources it. No image download/copy, package installation or GPU is required for replay. Inspect the printed notebook working directory and run batch commands there, not in the separate instructor preflight directory.

```bash
sbatch cpu.sbatch
# After it completes, inspect the raw results and workflow_manifest.json.
sbatch container.sbatch
# Where a working compatible MPI stack is available:
sbatch hybrid.sbatch
# Optional C-in-container extension, requiring a gcc/OpenMP image:
sbatch container_c.sbatch
```

CPU/hybrid allocations request one node, four CPUs total, 1 GB and six minutes. Container replay uses one CPU, 256 MB and three minutes. Keep compute jobs separate from the Jupyter kernel. Queue wait is additional. Submission is not PASS.

The CPU stage writes actual reference/candidate files and a hash manifest. The replay stage refuses stale source/inputs, compares host versus container numerical results, records environment differences, explicitly forwards a sentinel variable, and expects rejection of a deliberately corrupted copy. Read-only input and writable output binds are part of the experiment.

Native CPU and Apptainer replay are core activities. A missing image/runtime is NOT RUN for the required replay, not automatic completion. The instructor must verify image access before student release. MPI failure requires the actual diagnostic and the documented CPU route where applicable; no runtime result is invented. Spack and workflow tools are discussed by role; no Spack installation or advanced DSL syntax is required.

## Evidence and validation boundary

The notebook has 20 cells, 9 code cells; published outputs are cleared. Local tests executed all nine cells and actual-record analysis. Serial/OpenMP small/odd/default checks, an independent tiny Python reference, default repetitions and deliberate wrong-result rejection passed. New replay payload tests passed natively for top-1/top-k, ties, boundary cases, corrupted scores, source/input drift and CLI provenance.

Instructor-provided SciTech job **24943**, host **haskell**, completed with all attempted checks PASS: native GCC 14.2.0 serial build; independent correctness fixtures; checker negative tests; Q=32/Q=64 baseline workloads; guided OpenMP teams 1/2/4; guided MPI 2 ranks × 2 threads; and shared-image Apptainer replay including binds, environment forwarding and bad-result rejection. This is not validation of a student's assessed top-k solution.

The notebook's actual end-to-end SciTech prepare/submit/report/plot workflow still needs its own execution evidence. Published outputs remain cleared; missing student reports are NOT RUN. The notebook prints the exact terminal working directory, preserves an existing site_env.sh, and explains the CPU-before-replay dependency.


## Automatic student execution

Run the notebook cells in order. The submission/status cell defaults to `AUTO_RUN=True` and submits three bounded Slurm jobs. Dependencies serialize CPU, hybrid and container replay; replay requires CPU success but does not require MPI success. Job IDs are saved in `guided_jobs.json` and reused on reruns/kernel restarts from the same working directory. Source/configuration drift blocks reuse rather than silently submitting new jobs. There are no automatic retries.

The status cell waits up to 45 seconds, including queue time. If jobs remain PENDING/RUNNING, rerun that cell and the E4/E5 result cells later; no duplicate jobs are created. Numerical PASS comes from current reports, not scheduler state. If Jupyter lacks sbatch, use the printed SSH directory and manual commands instead; do not use both submission routes. A known old comment-only environment template is backed up and replaced; custom configurations are preserved.

Instructor notebook CPU job 24944 completed on SciTech with passing serial/OpenMP records. The new automatic orchestration is locally tested with simulated scheduler responses; actual automatic end-to-end SciTech execution is still to be tested.
