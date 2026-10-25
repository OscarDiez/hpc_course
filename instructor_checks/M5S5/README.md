# M5S5 instructor cluster preflight

This standalone launcher checks the real assignment baseline/checker and guided C code in a separate Slurm allocation. It embeds snapshots of those files with their Git blob IDs and records SHA256 hashes. It does not download code in the compute job or install packages, and supplies no student parallel/optimized top-k solution.

## Run on the SciTech login node

```bash
curl -fL 'https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s5-vector-search-2026/instructor_checks/M5S5/verify_cluster.sh' -o m5s5_check.sh
bash m5s5_check.sh
```

It creates a new owned `m5s5-preflight.XXXXXX` directory and submits one CPU job: one node, four CPUs total, 1 GB, six minutes. Queue waiting is additional. The launcher prints the job ID, log path and exact summary command. Submission alone is not PASS. `SUMMARY.txt`, `report.json`, raw result files and `commands.log` appear in that folder as the job runs.

If your site toolchain needs module commands, put the exact verified commands in your existing `site_env.sh` and use:

```bash
M5S5_SITE_ENV="$HOME/path/to/site_env.sh" bash m5s5_check.sh
```

The path above is a placeholder. The launcher copies that file to the new owned folder and sources it in the compute job. It does not guess module names. The current environment is used when no file is provided; availability is checked. No account or QoS is added.

For the optional container check, export `M5S5_IMAGE` to the absolute path of an existing trusted C/OpenMP-capable SIF before launching. No image is pulled. If your site specifically requires a confirmed `srun --mpi` setting, export that exact setting in `M5S5_SRUN_MPI`; otherwise the site default is used.

## Checks

- Serial build with the current compiler.
- Independent Python oracle for all ordered top-k results: normal, exact ties, N=1 and K=N fixtures.
- Checker rejection of wrong last neighbour, duplicate ID, missing query, wrong score and changed problem metadata.
- Three real main-workload baseline repetitions at Q=32 and three sensitivity repetitions at Q=64; all result lists checked for repeat consistency.
- Real guided C OpenMP code with actual teams of 1, 2 and 4 threads, three repetitions each, checked against serial output.
- Where available, the real guided MPI/OpenMP code with two ranks and two threads, checked against serial output.
- Where configured, compilation inside an existing Apptainer image and four-thread execution with numerical comparison and image hash.

## Interpret results

PASS: the named executed check succeeded. FAIL: an attempted check failed; inspect its diagnostic and commands.log. SKIPPED: an optional feature was unavailable/unconfigured. NOT RUN: a prerequisite prevented execution or the run is explicitly local. A missing optional feature does not prove that MPI/container execution works. A failing optional MPI launch needs correction or the assignment's documented CPU alternative.

Successful baseline and OpenMP checks support releasing the CPU route. MPI and image results determine which optional routes have actually been validated. The script does not validate an as-yet unwritten student implementation, speedup claim, PDF submission or Blackboard settings. It cannot certify every student's account permissions or resource quota; a short student-account pilot is still useful.

If the job produces no summary, inspect the printed Slurm log and run `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocCPUS`. A module-loading failure or scheduler termination can happen before Python starts. Cancel an unwanted job with `scancel JOBID`.

Local authoring test: all CPU/checker tests passed, including real default-size runs. Slurm, MPI and Apptainer were unavailable locally. **SciTech execution is NOT RUN until you execute the launcher there.** `bash m5s5_check.sh --local` is an explicit instructor workstation mode and must never be used to benchmark on a cluster login node. Every report is marked LOCAL or SLURM.
