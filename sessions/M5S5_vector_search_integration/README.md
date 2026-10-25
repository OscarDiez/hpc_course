# M5S5 · Vector search integration lab

Guided C/Slurm lab consolidating Module 5, with a required Apptainer result-validation workflow. The individual cluster assignment, module quiz and group milestone are separate activities.

[Notebook](M5S5_vector_search_integration.ipynb)

Raw URL for JupyterLab **File → Open from URL**:

https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s5-vector-search-2026/sessions/M5S5_vector_search_integration/M5S5_vector_search_integration.ipynb

Save your own copy and run the Python 3 cells in order. The notebook embeds all C, Python, shell and batch files and writes a new owned work folder. No package installation or image download is performed.

## Module 5 practice map

| Earlier session | Actual practice here |
|---|---|
| M5S1: hybrid patterns | MPI shards, local OpenMP work, equal four-CPU layout comparison where site MPI works |
| M5S2: scalable applications | Independent correctness fixture, actual team counts, repeated timings, communication/selection evidence |
| M5S3: portability/reproducibility | Inspect modules/wrapper paths; execute Python result validation inside an existing Apptainer image; binds, clean environment, explicit variable forwarding, image/input/script hashes |
| M5S4: project workshop | Keep a runnable, traceable compute-to-validation workflow and explain limitations for a project handover |

The C search uses deterministic synthetic vectors; it does not evaluate semantic relevance. Native C outputs are checked against serial/independent references. The core container stage replays validation and analysis of these files; it does not claim C execution inside the Python image.

## Run

Use the [SciTech guide](../../cluster/README.md). Work under your shared home directory, not a compute node's `/tmp`. Edit `site_env.sh` with the actual compatible module commands and the approved existing **Python 3 SIF** path in `M5S5_IMAGE`.

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

Instructor-provided SciTech job **24922**, host **haskell**, passed native serial build, independent fixtures, checker negative tests, Q=32/Q=64 baseline workloads and guided OpenMP teams 1/2/4. Its MPI link failed with mixed EESSI/system OpenMPI paths. This is evidence for those core C computations, not a successful MPI result. The newly revised manifest/container workflow still requires a fresh site run. **Actual Apptainer binds, clean environment and forwarding: NOT RUN until site execution evidence exists.** A real approved Python SIF path is required; none is guessed here.
