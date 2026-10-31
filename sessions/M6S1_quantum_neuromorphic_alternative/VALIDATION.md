# M6S1 validation · 31 October 2026

## Completed locally

- Notebook JSON/cell structure and unique cell IDs checked. All code parsed and all 17 code cells executed in order, with stdout saved into the notebook. This used a Python execution harness, not a live Jupyter kernel/frontend.
- Python 3.12.14, GCC 13.3.0, Linux x86_64. Compile: `gcc -O2 -std=c11 -Wall -Wextra -Werror m6s1_lab.c -lm -o m6s1_lab`.
- E1: both algorithms agree at every frame and checksums match independent Python calculation. Period 50 gives 20 event updates and period 100 gives 10; dense reference performs 16,000 in both cases. Input streams differ across periods, so cross-period checksums are not expected to match.
- E2: independent recurrence check. Threshold 1.0 produces two spikes in twelve steps; 1.3 produces one. Thresholds 0.1, 1.6 and 2 also checked.
- E3: independent matrix reference and quantization bound. At three bits the maximum output error is approximately 0.024369747899; eight bits represents this constructed matrix exactly because 255 is divisible by 17. Two- and twelve-bit bounds also checked. This does not establish generally exact eight-bit inference.
- E4: normalization and analytic `P(1)=sin²(phi/2)` checks at 0, pi/2, pi, 1 and 2*pi. Seeded CPU sampling distinguishes exact probabilities from sampled frequencies. Storage figures are calculated, not allocated.
- E5: assumed cost model gives 33 ms with 5 ms overhead and 108 ms with 80 ms overhead, against an assumed CPU baseline of 100 ms. These are calculated examples, not timings.
- Optional OpenMP: actual local two-thread matrix calculation agrees with serial. No speed/scaling claim.
- Six invalid C invocations rejected. Missing compiler reports SKIPPED. Attempted failed compilation reports FAIL.
- Mocked scheduler tests checked inherited memory cleanup, saved-job reuse and missing result NOT RUN. These tests never contacted Slurm.
- `bash -n` passed for the batch script. Tables and generated Matplotlib plot inspected. The plot has explicit CPU simulation/model labels.

## Not executed

- Actual SciTech Slurm submission, queue state, accounting, module loading and allocated compute-node execution: NOT RUN.
- JupyterLab URL import and browser rendering: NOT RUN. Notebook uses standard nbformat 4.5 structure, Python 3 cells and PNG output.
- GPU, MPI and specialized QPU/neuromorphic/photonic/resistive hardware: SKIPPED by design.

## Instructor SciTech acceptance

Open the raw notebook in SciTech, save a personal copy, restart the kernel and run in order. Record the real job ID and log. Rerun collection after the job finishes. Require E1–E5 PASS, context SLURM_CPU, a matching source hash/settings and the matching submitted job ID. Inspect any FAIL/SKIPPED rather than treating it as success. Change one parameter, repeat and save the new records. No SciTech acceptance is implied by this report.
