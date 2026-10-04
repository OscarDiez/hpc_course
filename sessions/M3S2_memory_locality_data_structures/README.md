# M3.S2 — Memory, locality and data structures

Notebook: [open the raw notebook](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M3S2_memory_locality_data_structures/M3S2_memory_locality_data_structures.ipynb)

## Start in class

Save a personal copy. Use Python 3 and **Restart Kernel and Run All Cells**. Section 1 writes the tracked C/sbatch sources into `m3s2_demo_files_v2`. Section 2 submits one CPU job, waits for complete results and stops with recovery instructions if the job fails or the queue takes over ten minutes. The eight-minute execution limit is separate from queue waiting. Requests: one node, 16 CPUs, 4 GB, CPU partition, `foss/2023b` when environment modules are available.

An unchanged configuration reuses its saved job even after a kernel restart. Changed settings generate a new job only after the previous job leaves the queue. For deliberate repeated measurements set `REPEAT_SAME_EXPERIMENT=True`, run the submission cell, then reset it to False. Saved experiment JSON files contain settings and raw output. GPU experiments and inter-node MPI timings are outside this notebook.

## Suggested 80-minute route

- 0–10: predict, inspect host versus allocation, configure and submit.
- 10–30: latency/bandwidth, stride, row/column traversal.
- 30–50: false sharing, particles in AoS/SoA, qualitative Roofline.
- 50–65: choose image tiling, climate fusion or sparse interactions; change one setting.
- 65–80: compare two experiments and complete the exit ticket.

NUMA, compiler reports, unrolling and prefetching are extensions. All native modes run in the same job, so their output is available without separate submissions. The final small Python activities work anywhere and validate results rather than claiming hardware speedup. Plots require Jupyter/IPython, but no Matplotlib or package installation.

## Experiments students can change

| Problem | Control | Evidence |
|---|---|---|
| Streaming arrays and particle fields | `M3_ARRAY_N`, `M3_THREADS` | Useful throughput, layout/runtime differences |
| Image transpose | `M3_MATRIX_N`, `M3_TILE` | Runtime, all-cell correctness including edge tiles |
| Climate preprocessing | `M3_ARRAY_N` | Two-pass versus fused runtime and equivalent output |
| Private worker counters | `M3_THREADS`, `M3_COUNTER_ITERS` | Packed/padded runtime and correct counts |
| Sparse interaction matrix | `M3_SPARSE_N`, `M3_NNZ_PER_ROW` | Storage estimate, real SpMV, serial reference |
| Prefetch | `M3_PREFETCH_DISTANCE` | Measured improvement, no change or slowdown |
| Unrolled vector addition | Non-multiple-of-four `M3_ARRAY_N` | Tail correctness and compiler-dependent runtime |

Actual C functions are displayed beside interpretations, including OpenMP directives. Toggle `SHOW_C_CODE` off to shorten output. Embedded notebook sources and files under `session_demos/12_memory_locality` must remain identical when editing.

## Measurement limits

The pointer chase is latency-sensitive but can fit in cache; it is not a certified DRAM-latency test. Stride uses volatile accesses and a scalar accumulation, deliberately constraining optimisation. Useful streaming bytes are not hardware-measured DRAM traffic. The Roofline compute ceiling and some intensities are illustrative. Padding assumes 64-byte cache lines. NUMA uses fresh anonymous mappings and bound workers, but does not measure page residency or guarantee that the allocation spans memory domains. Some kernels use short fixed-order measurements: repeat jobs before attributing small differences to an optimisation. Cache-miss counters and energy are not measured. Sparse storage counts stored entries (including possible duplicate columns); it never allocates the huge dense equivalent.

## Validation (2026-10-04)

All 12 C modes compiled and ran locally with full-result checks for key transformations, a 257-wide matrix with partial tiles, and 1,048,579 array entries to exercise the unrolled tail. Invalid inputs were rejected. All notebook cells ran in model mode and analysis cells processed actual native output; generated SVG plots parsed successfully. Simulated Slurm checks covered submission, unchanged/restarted reuse, changed settings, active-job guards, timeout and failed-job handling. This does not replace a successful SciTech job/export before teaching.
