# M5S3 validation · 24 October 2026

Build: `2026-10-24-v2`. Author environment: Linux x86_64, Python 3.12.14, GCC 13.3.0. These are **local author results**, not measurements from SciTech. Three fresh working directories exercised default inputs, threshold 22 C, and E1 input 0 C.

| Check | Result | Scope |
|---|---|---|
| Notebook JSON/cell structure, unique IDs, Python syntax | PASS | nbformat 4.5 structure checked; all 20 code cells parse |
| Sequential execution of every Python code cell | PASS | Python harness; not an interactive Jupyter kernel test |
| Unmodified application, changed teaching dependency | PASS | Two actual fresh processes; 20 C gives 293.15 K with v1 |
| E1 input changed from 20 C to 0 C | PASS | Parameter-aware assertions still hold |
| Child environment and unchanged parent | PASS | Actual process environment |
| Empty venv without pip/packages | PASS | Actual subprocess reports a separate prefix |
| Module inspection | SKIPPED locally | Module shell function unavailable; SciTech must record its own environment |
| Temperature baseline and repeated run | PASS | 1,000 readings; mean 20.035621014 C; fraction above 25 C = 0.253 |
| One parameter changed to 22 C | PASS | Same input identity and mean; fraction = 0.403 |
| C11 serial compilation/run | PASS | Mean 20.03562101400004 C, baseline fraction 0.253 |
| C11 OpenMP compilation/run | PASS | **Two actual threads**; mean within 1e-12 tolerance; baseline fraction 0.253 |
| Original/frozen CSV byte identity | PASS | Same SHA-256 in notebook workspace and frozen run |
| Cache invalidation | PASS | Observed COMPUTE, REUSE, COMPUTE |
| Independent replay | PASS | Separate process, no notebook variables |
| Tampered input rejected before calculation | PASS | Separate damaged copy exits 2 with integrity mismatch |
| Handover ZIP extracted and independently replayed | PASS | Fresh extracted directory |
| Generated Slurm shell syntax | PASS | `bash -n`; allocation/step both one task × two CPU slots |
| SciTech Slurm submission and compute-node execution | NOT RUN | No live SciTech access used |
| Apptainer execution | SKIPPED locally / NOT RUN on SciTech | No prepared SIF/runtime execution; no fabricated result |
| Spack concretization/install | NOT RUN | Specification generated only |
| MPI/GPU and workflow engines | NOT RUN | Conceptual coverage; no execution claim |

A job-wide PASS means no attempted case failed. It does not change an individual SKIPPED case into PASS. Numerical agreement with a reference checks implementation consistency; it does not establish that synthetic sensor data constitute a validated physical model.

## Instructor acceptance on SciTech

1. Open the branch notebook in the intended Jupyter kernel; run it from a personal shared directory.
2. Verify the `cpu` partition, one-task/two-CPU allocation, module status and compiler version in that job's log.
3. Collect the matching complete report. Confirm `python`, `c_serial` and `c_openmp` PASS; actual OpenMP thread count 2. Preserve any genuine capability SKIPPED or execution FAIL.
4. When a trusted prepared SIF is available, set `RUN_CONTAINER=True` and its absolute path, run a new fingerprint after the prior job finishes, and require an actual container PASS before showing a container result in class.
5. Save the executed notebook and real `.out`, `.err`, settings and report as SciTech evidence. Until then the slide plan must label this demonstration pending or optional.

No notebook benchmark timings or hypothetical container results are published. The small input teaches reproducibility rather than speedup.



## v3 · Existing-image container launches (24 October 2026)

Build: `2026-10-24-v3-existing-containers`. Container execution is enabled by default. One bounded CPU batch now discovers course images and executes environment, bind/output and scientific replay examples, plus a NumPy stiffness solve when available. No download or installation occurs.

### New local checks

- All **21 Python code cells** parse and execute sequentially with `RUN_CLUSTER=False` in the test harness only; the student notebook retains True.
- Native worker Python, serial GCC and actual two-thread OpenMP: PASS.
- Environment, CSV read/write and NumPy stiffness/load-scaling payloads: PASS on the local host.
- CLI orchestration: PASS under an explicit host-process test double. This verifies command construction, result checking and report handling, **not an actual container**.
- Existing-image directory inventory, paths with spaces, explicit missing image, incompatible-image failure, fallback candidate, expected-hash rejection, half-load parameter and extracted-ZIP replay: PASS.
- Generated Slurm script: `bash -n` PASS. Allocation remains one task × two CPU slots, 1 GiB, three minutes.
- Actual Apptainer/SIF execution for this revision: **NOT RUN locally and not yet evidenced on SciTech**. Do not treat the test double as a container measurement.

See `validate_notebook.py` and `v3_validation_results.json` for reproducible checks and exact scope.

### Instructor-provided cluster result: prior v2 native path

The newly supplied HTML export contains a complete real report for job **24608** on **julia**, with `foss/2023b` loaded and GCC **13.2.0**. It confirms Python, serial C and OpenMP PASS, with **two actual OpenMP threads**. The old container case is SKIPPED because `RUN_CONTAINER=False`. The report is preserved verbatim inside `scitech_native_evidence.json`; its date of execution is not independently inferred from the export.

These native results supersede the earlier statement that no SciTech native evidence exists. They do not validate the changed v3 helper or any container path.

### Accept the new container route on SciTech

1. Open the updated raw branch notebook and save a fresh student copy. Keep `RUN_CONTAINER=True`.
2. Inspect E4a's inventory. If images are elsewhere, set `IMAGE` or the instructor-approved `IMAGE_DIRS`.
3. Wait for earlier jobs to finish, then run preparation/submission once. Collect the new job's matching complete report.
4. Require actual PASS for `container_image`, `container_environment`, `container_bind` and `container`. NumPy may be SKIPPED if absent; require its own PASS before demonstrating the solve.
5. Save the executed notebook/report/logs. Pin the accepted image path and hash for a controlled rerun. Preserve any genuine FAIL or SKIPPED result.
