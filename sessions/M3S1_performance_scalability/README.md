# M3.S1 — Performance Models, Scalability & Parallel Patterns

Hands-on notebook for Session 11 of the 2026 Bachelor HPC course. Open the existing slide URL:

https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M3S1_performance_scalability/M3S1_performance_scalability.ipynb

## Learning activities

Students predict, run, change one variable, and explain their own results.

| Slide topic | Notebook activity | Evidence |
|---|---|---|
| Speedup, efficiency, measurement boundaries | Sections 1–3: timed OpenMP/MPI phases, variability and core-seconds | Real Slurm job |
| Amdahl, diminishing returns | Sections 5–6: serial/parallel timings, editable serial fraction, overhead sandbox | Real timings + labelled model |
| Gustafson | Section 6: fixed vs scaled workloads and serial-fraction conversion | Model |
| Strong/weak scaling | Section 7: fixed N vs constant N per worker, editable work size | Real OpenMP |
| Map/Reduce | Sections 4 and 11: array work and global sums, correctness across worker counts | Real OpenMP/MPI |
| Scan | Section 12: sensor record offsets and block scan against sequential reference | Small correctness activity + real OpenMP |
| Stencil, decomposition, halo exchange | Section 13: heated rod, deliberately omitted halos, MPI field validated against serial reference | Small numerical activity + real MPI |
| Task farm / manager-worker | Section 14: unequal simulation cases, load model and static/dynamic schedules | Model + real OpenMP |
| Pipeline/dataflow | Section 15: editable stage durations, batch latency, steady-state throughput | Sleep-based model |
| Search and divide-and-conquer | Section 16: array search, recursive merge-sort leaves, BFS friendship frontiers | Real OpenMP + small correctness activities |
| Embarrassingly parallel / Monte Carlo | Section 18B: π, sample count, seed, uncertainty versus worker count | Real OpenMP |
| Resource decision | Final challenge: deadline, runtime and active-worker seconds | Student's measurements |

## Classroom use (80 minutes)

Start with predictions and submit the default CPU job. While queued, run the Amdahl/Gustafson models. Collect results separately, then rerun the analysis cells. Prioritize strong/weak scaling, serial phases, the rod/halo activity, and one student-selected controlled investigation. Pipeline, sorting, GPU execution and additional pattern investigations can be completed afterwards.

`RUN_CLUSTER=False` and `RUN_GPU_EXTENSION=False` initially. Setting the CPU flag to True submits a job after configuration; the separate collection cell never submits. Recheck a queued job instead of submitting duplicates. A completion marker is required before loading measurements. Each completed CPU experiment saves a JSON record containing job ID, configuration and raw output.

## Editable experiment settings

The Section 2 settings cell exposes total items, work per item, work per thread, small problem size, serial iteration percentage, heavy-task weight, stencil size/steps, MPI items, Monte Carlo samples and seed. Keep all but one setting fixed when comparing jobs. Default files download into `m3s1_demo_files_v8/`; source hashes detect stale reference versions. Local edits are retained and announced.

## Resources and dependencies

- Python 3 standard library is sufficient. Matplotlib is optional; tables work without it.
- Models and small correctness activities work without Slurm. Real benchmarks never run automatically on a login/Jupyter node.
- CPU job: one node, 16 CPUs, 4 GB, CPU partition, six-minute execution limit, `foss/2023b`.
- MPI uses explicit job steps of up to eight ranks on that same node, one CPU per rank. It does not test inter-node network scaling.
- OpenMP strong/weak and Amdahl have warm-ups and three repetitions. Other pattern timings are exploratory and should be repeated for reliable comparisons.
- The arithmetic map is a synthetic scalability kernel. The Monte Carlo, diffusion, prefix and graph activities give concrete problems without claiming production realism.
- Monte Carlo sample indices and seed define the same sample set across thread counts. Fixed N changes runtime, not the estimate. Approximate standard error assumes suitably independent uniform samples.
- Full MPI stencil validation occurs outside the timing interval. Independent maxima of compute and communication times need not add to maximum total time.
- Active-worker seconds are a resource proxy. The job reserves 16 CPUs throughout; requested allocation cost, electricity use and billing can differ.

## Validation

The revised notebook's code cells were executed in order without Slurm; measured OpenMP analysis paths were also exercised with locally generated output. All ten OpenMP modes compiled with GCC `-O3 -fopenmp -std=c11 -Wall -Wextra` and ran. Checks covered invariant Monte Carlo counts, map sums, block scan (including empty blocks), rod partition correctness, deliberate missing-halo error, BFS distances and recursive sorting. The MPI stencil's numerical partition logic was compared with a serial reference in a Python simulation.

Actual SciTech Slurm scheduling, multi-rank MPI transport and optional GPU execution still require a run on SciTech. A successful local check is not a claim that those cluster services were tested.
