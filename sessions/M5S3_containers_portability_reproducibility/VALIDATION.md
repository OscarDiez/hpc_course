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
