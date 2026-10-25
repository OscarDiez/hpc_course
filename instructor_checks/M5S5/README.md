# M5S5 instructor cluster preflight

Standalone bounded Slurm test of the supplied assignment baseline/checker and guided code. No student optimized top-k answer is supplied. Sources are embedded with recorded identities; the compute job installs/downloads nothing.

On SciTech, work from a shared home directory:

```bash
curl -fL 'https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s5-vector-search-2026/instructor_checks/M5S5/verify_cluster.sh' -o m5s5_check.sh
bash m5s5_check.sh
```

Configure `M5S5_IMAGE` to the teacher-approved existing **Python 3 SIF** before running. GCC inside that image is no longer needed for the core container check. If modules must be loaded in the batch shell, set `M5S5_SITE_ENV` to the absolute path of your existing `site_env.sh`. The launcher copies and sources it; it never guesses module names. `M5S5_SRUN_MPI` may name a site-confirmed launcher setting where required. No account/QoS is added.

The new owned report directory and job/log paths are printed. Resources: one node, four CPUs, 1 GB, six minutes. Queue wait is additional. Read SUMMARY.txt and report.json only after state COMPLETED; RUNNING/IN PROGRESS is not final success. Missing output may indicate an early module failure or scheduler termination: inspect the log and `sacct`.

## Checks

- Real serial build and independent normal/tie/N=1/K=N numerical fixtures.
- Checker rejection of five deliberate corruptions.
- Three main Q=32 and three sensitivity Q=64 serial runs.
- Real guided OpenMP code with actual 1/2/4-thread teams, three repetitions each.
- Available MPI: actual guided 2-rank × 2-thread launch/result check. A mixed compiler/system MPI link error is FAIL and needs a matching module stack or the assignment's documented CPU route.
- **Required Apptainer replay:** host Python and Python inside the existing SIF validate the same native C top-k outputs; read-only input/writable output binds; explicit environment forwarding; rejection of a corrupted copy; image/input/script identity. C computation remains native; C/MPI execution inside the Python image is not claimed.
- Optional C-in-image extension: set `M5S5_RUN_CONTAINER_C=1` only with a separately verified gcc/OpenMP image. This extension is not needed for the core workflow.

PASS means the executed check succeeded. FAIL means attempted failure. NOT RUN means no evidence; a missing required image keeps the overall result INCOMPLETE rather than silently passing. Optional unavailable MPI is SKIPPED. Any attempted failed check keeps the overall result FAIL. Inspect the named checks to distinguish validated CPU work from unresolved MPI/container capabilities.

Instructor-provided SciTech job 24922 on haskell passed the baseline, independent/checker fixtures, full-size workloads and guided OpenMP; it failed MPI linking with mixed EESSI/system OpenMPI paths. The revised Python-in-container replay still needs actual site execution with the configured image.

Local validation passed the replay payload for top-1/top-k, ties, boundary cases, corrupted outputs and CLI provenance. Manifest tests reject stale source/input hashes and local-only provenance in the required Slurm route. All nine revised notebook code cells ran locally. Local workstation mode (`bash m5s5_check.sh --local`) never claims SciTech execution and will remain INCOMPLETE for required Slurm/Apptainer capabilities. Never use local mode to benchmark on a cluster login node.

## Updated SciTech evidence

Instructor-provided job **24943** completed with all attempted checks PASS: serial build, independent/checker fixtures, main/sensitivity baseline workloads, guided OpenMP, guided MPI and shared-image Apptainer replay including binds, environment forwarding and deliberate-error rejection. Shared image: `/data/software/containers/pytorch-2.8.0-cuda12.8.sif`. The student starter's `site_env.sh` contains the tested native configuration. This supersedes the earlier unresolved replay status above. No student optimized top-k implementation is supplied or claimed validated.
