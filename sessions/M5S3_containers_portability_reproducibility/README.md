# M5.S3 · Containers, portability and reproducibility in HPC

IE University · Bachelor in Computer Science and Artificial Intelligence · 2026–27 · **Syllabus Session 23**.

[Notebook](M5S3_containers_portability_reproducibility.ipynb) · [Validation](VALIDATION.md)

## Open in SciTech JupyterLab

1. Open https://jupyter.iesci.tech:8085/ and select Python 3.9+.
2. **File → Open from URL…**, paste:

```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s3-reproducibility-2026/sessions/M5S3_containers_portability_reproducibility/M5S3_containers_portability_reproducibility.ipynb
```

3. Save a personal copy in a student-owned directory shared with compute nodes. Run cells in order. All payload sources are embedded; no checkout, pip installation or registry pull is required.

After merge, the corresponding `main` URL can be used. Until then the teaching branch contains the reviewed revision; `main` still contains the earlier notebook.

## Five practical experiments

| Experiment | Question | Actual execution | Main check |
|---|---|---|---|
| E1 · Dependency drift | Can unchanged app code give a different answer? | Two real Python child processes using small local teaching libraries | Celsius/Fahrenheit semantic difference detected |
| E2 · Process environment | Which interpreter and variables does a child use? | Process environment, real empty venv, read-only module probe | Child inheritance; parent unchanged; separate venv prefix |
| E3 · Recorded temperatures | What changes when the threshold changes? | 1,000-row CSV; Python mean/fraction; serial C and two OpenMP threads in the CPU batch | Input/code identities; repeat agreement; cross-language tolerance |
| E4 · Container boundary | Can a prepared SIF repeat the same calculation? | Optional actual Apptainer run in that same CPU batch | Same code/input hashes and numerical agreement; otherwise SKIPPED |
| E5 · Workflow and handover | Can another process replay and reject changed input? | Cache, manifest, independent process, damaged copy, handover ZIP | COMPUTE/REUSE/COMPUTE; replay PASS; intentional integrity failure detected |

Each uses problem → prediction → code → run → result → explanation → one-parameter experiment → takeaway. Six MCQs provide feedback after choosing A–D. Python prints a result table and an SVG chart from computed fractions; no plotting package is required.

## Class route (80 minutes)

Opening/setup 6; E1 8; E2 13; E3 17; E4 17; E5 13; recap/exit 6. Start the short CPU batch during E3/E4 preparation, then discuss container boundaries while it queues. If the job does not complete, preserve NOT RUN/QUEUED and use the validated local Python work without presenting it as SciTech evidence.

## SciTech execution

Read [cluster guide](../../cluster/README.md) and [infrastructure validation notes](../../cluster/COURSE_VALIDATION_NOTES.md). The notebook follows the existing EESSI/module initialisation route and attempts `foss/2023b`, recording what actually loaded. If modules are unavailable, the worker explicitly probes the active compiler and records its path/version.

- `RUN_CLUSTER=True` submits **one separate CPU batch** where `sbatch` exists. False opts out.
- Allocation: `cpu`, one node, one task, two CPU slots, 1 GiB, three minutes. The job step uses the same one-task/two-CPU shape, avoiding the earlier M5S2 per-task mismatch.
- Sources, input and settings are frozen in a fingerprinted run directory. `sbatch` runs with that directory as `cwd`, so `SLURM_SUBMIT_DIR` resolves the payloads correctly.
- Inherited Jupyter `SLURM_*` submission variables are removed from the child environment. Job ID reuse prevents duplicate submissions. An earlier active/unverified attempt blocks a new fingerprint.
- Rerun the **collection cell**, not submission, while queued. A report is accepted only with the matching job/run ID and completion marker. Inspect `.out`/`.err`; do not silently relabel failures.
- No GPU or MPI computation is needed or submitted. Their host compatibility requirements are conceptual reference material only.

## Optional container comparison

Use a prepared trusted Python 3 SIF with a compatible architecture and standard-library `csv`, `json`, `math`. Set `IMAGE` to its absolute shared path and `RUN_CONTAINER=True`, then rerun preparation/submission for a new settings fingerprint. No image location is assumed; the old sponsor paths and `gpu-node` partition examples are not SciTech defaults.

Missing runtime/image is SKIPPED. A runtime or numerical failure after attempting the calculation is FAIL. `--cleanenv --containall --bind RUN_DIR:/work --pwd /work` makes the path mapping explicit. Image hashes establish identity, not trust. The example definition uses a mutable teaching tag; pin and archive construction inputs and a trusted SIF before a lasting release.

## Evidence boundary

**Locally validated:** all 20 Python code cells, venv, dependency and parameter changes, generated GCC serial C and OpenMP (actual two threads), cache, independent replay, damaged-input rejection, replay from an extracted ZIP, and Bash syntax. See VALIDATION.md for exact values.

**Not validated on SciTech in this revision:** scheduler acceptance, shared-path visibility, actual EESSI/module setup, compute-node C/OpenMP run, and Apptainer execution. Spack concretization/install and actual workflow engines are NOT RUN. The clean student notebook contains no saved execution outputs. The author ran Python cells sequentially using a validation harness, not an interactive Jupyter kernel.

This revises the existing M5S3 notebook rather than replacing the legacy hybrid example at `chapters_examples/23.M5.S3.Hybrid_Computing(5_3).ipynb`.
