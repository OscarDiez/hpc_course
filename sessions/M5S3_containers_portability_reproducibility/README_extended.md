Extended reference for the previous complete M5S3 notebook. Start with [the shorter practical](M5S3_containers_portability_reproducibility.ipynb) for the modules and Apptainer exercise.

# M5.S3 · Containers, portability and reproducibility in HPC

IE University · Bachelor in Computer Science and Artificial Intelligence · 2026–27 · **Syllabus Session 23**.

[Notebook](M5S3_containers_portability_reproducibility_extended.ipynb) · [Validation](VALIDATION.md)

## Open in SciTech JupyterLab

1. Open https://jupyter.iesci.tech:8085/ and select Python 3.9+.
2. **File → Open from URL…**, paste:

```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s3-reproducibility-2026/sessions/M5S3_containers_portability_reproducibility/M5S3_containers_portability_reproducibility_extended.ipynb
```

3. Save a personal copy in a student-owned directory shared with compute nodes. Run cells in order. All payload sources are embedded; no checkout, pip installation or registry pull is required.

After merge, the corresponding `main` URL can be used. Until then the teaching branch contains the reviewed revision; `main` still contains the earlier notebook.

## Five practical experiments

| Experiment | Question | Actual execution | Main check |
|---|---|---|---|
| E1 · Dependency drift | Can unchanged app code give a different answer? | Two real Python child processes using small local teaching libraries | Celsius/Fahrenheit semantic difference detected |
| E2 · Process environment | Which interpreter and variables does a child use? | Process environment, real empty venv, read-only module probe | Child inheritance; parent unchanged; separate venv prefix |
| E3 · Recorded temperatures | What changes when the threshold changes? | 1,000-row CSV; Python mean/fraction; serial C and two OpenMP threads in the CPU batch | Input/code identities; repeat agreement; cross-language tolerance |
| E4 · Container boundary | Can a prepared SIF repeat the same calculation? | Actual Apptainer examples enabled by default in that same CPU batch; use existing local images | Same code/input hashes and numerical agreement; otherwise SKIPPED |
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

## Existing cluster containers: default classroom route

`RUN_CONTAINER=True` now makes the CPU batch **launch existing images**, rather than leaving the main topic disabled. No registry pull, build, pip/conda installation, GPU request or MPI run is needed.

1. Leave `IMAGE=""` to inspect `/data/software/containers` and the home-relative historical fallback `~/projects/def-sponsor00/shared/images`. The current shared CUDA image is documented in Practice 2, but earlier validation showed it was not accessible from every job. The old sponsor path also belongs to an earlier cluster setup. Both are checked at execution time, never assumed to exist.
2. If the course images are stored elsewhere, add that instructor-approved directory to `IMAGE_DIRS`, or set the absolute `IMAGE` path (also accepted through the `M5S3_IMAGE` environment variable).
3. Run **E4a inventory**, **preparation**, then **submission**. The existing guard prevents duplicate or overlapping unverified jobs. A previous completed native run has a different fingerprint and is not misrepresented as this new container run.
4. In the allocation, the worker checks compute-node visibility and tries up to four readable candidates. It probes `python3`, then `python`, for Python 3.9+ and standard-library support. The first compatible existing image is selected, and its full SHA-256/runtime version are recorded.
5. Rerun **collection** while queued. The actual report prints these cases:

| Case | Actual work and check |
|---|---|
| `container_image` | Launch Python inside the image, record its identity and runtime, enforce an optional expected SHA-256 |
| `container_environment` | Compare Python/executable metadata and the shared kernel/machine with the compute host |
| `container_bind` | Read the frozen host CSV through `/work`, write JSON inside the process, verify its contents from the host |
| `container` | Run unchanged `analysis.py`; require matching input/code hashes, threshold/count and numerical agreement |
| `container_numpy` | If NumPy is already installed, solve a 3×3 stiffness system, independently check residuals, double the load and verify doubled displacement |

The `examples/` directory contains exact readable mirrors of the four added Python payload/helper sources. The notebook embeds them, so students still need only the notebook file. The worker uses `--cleanenv --containall`, an explicit run-directory bind and two-thread OpenMP/BLAS limits. It uses the image's `python` when `python3` is absent.

**One-parameter task:** change `LOAD_SCALE` from 1 to 0.5 after the previous job finishes; predict half the displacement. Alternatively change only `THRESHOLD_C` to 22. Do not switch the image and scientific parameter simultaneously.

After a successful AUTO run, copy the report's path into `IMAGE` and its SHA-256 into `EXPECTED_IMAGE_SHA256` to pin a rerun. A checksum establishes byte identity, not trust in the image's origin.

No runtime or readable image means SKIPPED with diagnostics. Readable images whose attempted execution is incompatible produce FAIL with the attempted commands' diagnostics. Missing NumPy skips only that additional example. No completion report means NOT RUN/queued, never an invented success.

## Evidence boundary

**Locally validated:** all 20 Python code cells, venv, dependency and parameter changes, generated GCC serial C and OpenMP (actual two threads), cache, independent replay, damaged-input rejection, replay from an extracted ZIP, and Bash syntax. See VALIDATION.md for exact values.

**Instructor-provided SciTech evidence for the prior v2 notebook:** job **24608**, host **julia**, `foss/2023b` loaded, GCC **13.2.0**; native Python, serial C and actual two-thread OpenMP **PASS**. The container case was **SKIPPED** because the old default was disabled. See `scitech_native_evidence.json`.

**New v3 container examples:** local payload and orchestration checks passed, but actual Apptainer execution and the new automatic image route still require a live SciTech report. No simulated CLI test is presented as a container measurement. Spack concretization/install and actual workflow engines are NOT RUN. The clean student notebook contains no saved execution outputs. The author ran Python cells sequentially using a validation harness, not an interactive Jupyter kernel.

This revises the existing M5S3 notebook rather than replacing the legacy hybrid example at `chapters_examples/23.M5.S3.Hybrid_Computing(5_3).ipynb`.


## Maintainer validation

Run `python3 validate_notebook.py` locally to check syntax, all notebook Python cells, native C/OpenMP, inventory, pinned-hash rejection, unavailable/incompatible images, candidate fallback, bind outputs and replay. CLI orchestration tests use an explicitly labelled test double on the host; they cannot validate Apptainer isolation or a real SIF. Results are written to `v3_validation_results.json`.
