# M3.S3 — Profiling and Performance Analysis

Open `M3S3_profiling_performance_analysis.ipynb` in SciTech JupyterHub with Python 3, save a personal copy, then Restart Kernel and Run All.

Raw notebook URL:
https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M3S3_profiling_performance_analysis/M3S3_profiling_performance_analysis.ipynb

## What students actually do

1. Predict the effect of access order in a matrix analysis pipeline.
2. Run repeated, alternating comparisons and inspect median/range, whole-program and phase timings.
3. Locate the hotspot with gprof, then qualify what the profile can and cannot explain.
4. Compare checked identical output produced with small versus buffered writes; inspect strace summary/events when permitted.
5. Redistribute the same independent simulation tasks across OpenMP workers; inspect computation and barrier intervals and whole-team time.
6. Change one input, rerun and compare two saved experiments before writing an evidence-based claim.

Defaults: one Slurm CPU node, 4 CPUs, 4 GiB, 12-minute limit; 4096×4096 matrix, 3 traversals, 5 repeated A/B pairs. Sources and the runner are embedded, so there are no downloads or Python package installs. Serial measurements are pinned to one allowed CPU; OpenMP uses the allocation. Results are saved under `m3s3_lab/run-*/report.json`, with experiment records, source hashes and tool statuses. A persisted job ID prevents duplicate submissions after a kernel restart. A 10-minute collection timeout stops notebook execution; rerun the same collection cell later.

The Slurm script tries the existing EESSI foss/2023b, Score-P/8.4-gompi-2023b and CubeLib/4.8.2-GCCcore-13.2.0 modules. Adjust these names for another site. The native compiler is still usable if optional profiling modules are unavailable. Missing/restricted optional tools do not receive fabricated results.

## Coverage and limits

| Evidence | Status / interpretation |
|---|---|
| C timings and correctness | Required core experiments; any failure prevents CORE PASS |
| GNU time | Application child inside the compute step; not the srun launcher |
| gprof | Instrumented call counts plus sampled CPU time; serial program |
| strace | Optional real syscall counts and timestamped events; tracing overhead excluded from timing comparisons |
| perf | Optional sampling/counters; missing, restricted and unsupported events distinguished |
| Score-P / Cube | Optional real `.cubex` profiles; trace success requires an OTF2 anchor file |
| Memory checking | Optional Valgrind intentionally leaked/fixed 16 KiB allocation |
| Amdahl | Explicit model, compared with measured phase/total ratios |

This is not GPU profiling, network/MPI timing, measured energy, or durable parallel-filesystem throughput. The OpenMP ownership comparison illustrates critical-path and imbalance reasoning without claiming network communication. The pre-existing `m3s3_mpi_wait.c` remains a separate instructor extension and is not silently run in this single-task allocation.

I/O checks every output byte after timing; no fsync is performed. Matrix checks use an independent reference and floating-point tolerance. The OpenMP program verifies exactly-once task ownership and the summed result. Warmups are excluded from repeated comparisons. Small improvements need repetition; cache misses or DRAM traffic are not claimed without counter evidence.

## Validation of build M3S3-2026-10-04-v10

All 20 code cells ran locally with default C workload and an explicit local opt-in; all core correctness checks passed. Real gprof output identified `compute_bad` as the dominant function. All cells also ran in the no-cluster mode without invented measurements. Additional tests cover odd matrix dimensions, uneven task counts, partial final buffers, one-thread execution, invalid inputs, missing tools, and persistent-job collection/reuse. Slurm and unavailable/restricted profiler paths are checked with simulated responses; these are workflow tests, not cluster measurements. Actual SciTech execution must be verified from its exported notebook/report before claiming site-specific tool success or speedups.
