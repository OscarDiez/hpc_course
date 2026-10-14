# M5.S2 — Advanced MPI: scalable communication and process management (2026–27)

**Status:** New notebook, GitHub draft. MPI code has **not been executed on SciTech**. All results must come from real Slurm output; no canned benchmark values are included.

## Open the notebook
- [GitHub notebook on the teaching branch](https://github.com/OscarDiez/hpc_course/blob/teaching/m5s2-advanced-mpi-notebook-2026/sessions/M5S2_advanced_mpi/M5S2_advanced_mpi.ipynb)
- JupyterLab **File → Open from URL…**:
```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s2-advanced-mpi-notebook-2026/sessions/M5S2_advanced_mpi/M5S2_advanced_mpi.ipynb
```
- On approval and merge to main, replace the branch in this raw URL with `main`.

## Scope and conventions
Created in `sessions/M5S2_advanced_mpi/`, using the established 2026 `MxSy` convention and the previous **PREDICT → RUN → OBSERVE → EXPLAIN** method.

Five core MPI C examples: non-blocking halo point-to-point exchange; one-sided RMA with `MPI_Win_fence`; persistent request reuse; `MPI_Iallreduce`; 2×2 Cartesian topology with neighbors. Optional sixth: dynamic process spawn, disabled by default because cluster support is not assured.

## SciTech execution
See [cluster/README.md](../../cluster/README.md) and [M2.S3 MPI lab](../M2S3_mpi/README.md). In IE JupyterLab open the notebook with a Python 3 kernel, save a personal copy, review the configuration cell and check `sinfo`. Set `RUN_CLUSTER=True` only when ready. The notebook generates a personal `m5s2_demo_files/` working folder, writes real C sources and one Slurm batch script, and asks for **one node, four MPI tasks, 1 GiB and five minutes**. The default partition is `cpu` and optional `foss/2023b` module; confirm both. `sbatch` records actual stdout/stderr with structured PASS/FAIL/SKIPPED results. The same fingerprint reuses its job ID to prevent accidental duplicate submissions. Changing a setting generates a different fingerprint but does **not** submit if the previous job is still queued/running. Spawn remains off unless separately enabled and validated.

**No MPI jobs run on the login node. No GPU or pip installs.** If SciTech is unavailable, only the independent small Python reference checks run; MPI results remain NOT RUN. Earlier M2.S3 testing observed inconsistent inter-node MPI support, so these demos stay on one node.

## Time plan (~80 minutes)
0–10 introduction, infrastructure and prediction; 10–25 non-blocking point-to-point; 25–40 RMA; 40–50 persistent; 50–60 non-blocking collective; 60–70 topology; 70–80 comparison and exit question. Dynamic spawning is optional backup/teacher discussion.

## Validation
Notebook JSON structure and content can be verified without MPI; C compilation, MPI correctness and measurements require an MPI compiler and runtime. **No SciTech execution evidence is claimed.** If available, report job ID, compiler/module, output logs and exact source revision after the live validation.

**Official syllabus:** The [IE-published 2025–26 syllabus](https://docs.ie.edu/university/Grados/BCSAI/HIGH_PERFORMANCE_COMPUTING.pdf), Session 22 (5.2 Advanced MPI), explicitly includes dynamic process management, one-sided communication, persistent requests and guided programming exercises. The **2026–27 final syllabus remains unverified**; confirm before slide approval. Non-blocking collectives and Cartesian topology are supporting extensions rather than mandatory syllabus core topics.

## Five one-parameter classroom exercises
In the notebook **configuration cell**, change **one** option, then run the source-preparation and submission sections again. Do not only hand-edit an already-generated `.c` file: the notebook regenerates it from the configured `PROGRAMS` sources.

| Setting | Default | Prediction and verified expectation |
| --- | ---: | --- |
| `OVERLAP_WORK_ITERS` | 400000 | Increase/decrease independent CPU work; **no guaranteed speedup**, verify received payloads |
| `RMA_WRITTEN_VALUE` | 42 | Change target payload; expect the same value after the closing fence |
| `PERSISTENT_STEPS` | 4 | Reuse requests for more/fewer steps; all exchanged values must match |
| `TOPOLOGY_PERIODIC` | False | True adds horizontal wrap-around neighbours |
| `IREDUCE_RANKS` | 4 | Set 2 to predict global sum 3 instead of 10 |

The notebook builds an **observed-only** topology plot if real rank coordinates are present and Matplotlib is already available. It does not install plotting packages or fabricate results.