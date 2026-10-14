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
See [cluster/README.md](../../cluster/README.md) and [M2.S3 MPI lab](../M2S3_mpi/README.md). In IE JupyterLab open the notebook with a Python 3 kernel, save a personal copy, review the configuration cell and check `sinfo`. Set `RUN_CLUSTER=True` only when ready. The notebook generates a personal `m5s2_demo_files/` working folder, writes real C sources and one Slurm batch script, and asks for **one node, four MPI tasks, 1 GiB and five minutes**. The default partition is `cpu` and optional `foss/2023b` module; confirm both. `sbatch` records actual stdout/stderr with structured PASS/FAIL/SKIPPED results. The same fingerprint reuses its job ID to prevent accidental duplicate submissions. Spawn remains off unless separately enabled and validated.

**No MPI jobs run on the login node. No GPU or pip installs.** If SciTech is unavailable, only the independent small Python reference checks run; MPI results remain NOT RUN. Earlier M2.S3 testing observed inconsistent inter-node MPI support, so these demos stay on one node.

## Time plan (~80 minutes)
0–10 introduction, infrastructure and prediction; 10–25 non-blocking point-to-point; 25–40 RMA; 40–50 persistent; 50–60 non-blocking collective; 60–70 topology; 70–80 comparison and exit question. Dynamic spawning is optional backup/teacher discussion.

## Validation
Notebook JSON structure and content can be verified without MPI; C compilation, MPI correctness and measurements require an MPI compiler and runtime. **No SciTech execution evidence is claimed.** If available, report job ID, compiler/module, output logs and exact source revision after the live validation.

**Official syllabus:** The official 2026–27 IE teaching plan was not among materials accessible during creation. This draft follows the supplied M5.S2 Advanced MPI slides and legacy notebook. Validate the module learning outcomes against the official teaching plan before final slide approval.
