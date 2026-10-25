# M5S5 · Individual cluster assignment
## Batched top-k vector search: parallelize, diagnose and justify

**Individual work.** This is separate from the guided notebook, the Module 5 quiz and the group project milestone. Submit your own cluster evidence and explanations. Deadline and course-grade weight are those announced in Blackboard; the 100 points below describe this assignment's internal marking only.

## 1. The application

A retrieval service receives a batch of query vectors. For each query, it must return the K collection items with the largest cosine similarity. This is the numerical search operation used in recommendation and retrieval pipelines. The supplied deterministic vectors stand in for embeddings: you are evaluating computational behaviour, not semantic relevance or a language model.

You receive a correct, deliberately simple **serial C baseline**. It stores normalized float vectors, accumulates dot products in double precision, sorts every collection item for each query, and returns the ordered top-k. You must develop your own parallel version, measure it through Slurm, investigate a bottleneck and implement one further optimization. A speedup is not guaranteed and is not itself the grading target.

This extends the guided top-1 example: a correct local winner is insufficient for a complete, ordered top-k. Merely changing the guided program's name or repeating its timings does not complete this task.

## 2. Starter files

Use `topk_baseline.c`, `check_topk.py`, `baseline.sbatch` and `site_env.sh` from this assignment directory. Preserve the original baseline. Put your implementation in `topk_candidate.c` and record the source hashes and starter repository commit.

The supplied batch script runs only three serial baseline repetitions. **You must add your own candidate compilation, launches, result files and checks.** Inspect and adapt the script; do not label the starter's serial results as parallel execution.

Load the real SciTech compiler modules in `site_env.sh`. Use the course cluster guide for access. All significant runs must use a compute allocation via Slurm, normally `sbatch` and `srun`. Do not benchmark on the login node or an ordinary Jupyter kernel. Do not install packages, MPI stacks or containers.

## 3. Fixed problem and correctness contract

Main workload: **N=65536 collection vectors, D=128 dimensions, Q=32 queries, K=10**. Use the supplied generator and IDs unchanged. The generator depends on global IDs, so MPI partitioning must not regenerate different data. N, D, Q, K and the tie-fixture flag are recorded in each JSON result.

For **every query**, return exactly K unique valid IDs, ordered by decreasing computed score; break exactly equal scores by increasing global ID. Use the baseline's float input storage and double accumulation. Scores must agree with the serial reference with relative and absolute tolerance 1e-10. Approximate search, reduced precision and dropping queries are outside this assignment.

Preserve the baseline JSON keys. Set `ranks` and `threads` to the actual runtime counts, not merely requested values. Retain the complete `hits` array. You may add useful timing and provenance fields. If parallel phases overlap, define your phase timings rather than asserting their sum equals elapsed time.

Required checks, before accepting performance:

1. Normal small case: `N=19 D=7 Q=3 K=5`.
2. Exact-tie case: the same arguments plus `ties`. All collection rows are identical; each query must return IDs 0,1,2,3,4 in that order.
3. Boundary case: `N=1 D=7 Q=1 K=1`.
4. Every main-workload and sensitivity run, including every query and all K neighbours.

The optional argument `ties` must retain its baseline meaning. Examples:

```bash
./topk_baseline 19 7 3 5 > reference-small.json
./topk_candidate 19 7 3 5 > candidate-small.json
python3 check_topk.py reference-small.json candidate-small.json
./topk_baseline 19 7 3 5 ties > reference-ties.json
```

Run these commands within the appropriate Slurm step for the implementation being tested. Keep reference files for each workload. A checker PASS validates the returned top-k, not the scientific relevance of synthetic embeddings or the reported runtime.

## 4. Required experiments

### A. Reference and parallel scaling

Measure the supplied serial baseline and your first correct OpenMP version with **1, 2 and 4 threads** on the main workload. Use three independent process runs per configuration; retain each elapsed time and report the median and range. Keep input, timing boundaries, compiler optimization policy and algorithm fixed within the OpenMP scaling series. Compare OpenMP one-thread time with the separately compiled serial reference to expose implementation overhead.

Your OpenMP decomposition is your decision. Explain which loop/work units are parallel, which data are shared/private and how you avoid races. Setting `OMP_NUM_THREADS` without adding parallel work does not count.

### B. One justified optimization

Use phase instrumentation or an available profiler to support one bottleneck hypothesis. Then change one meaningful aspect **beyond adding OpenMP**, such as selection strategy, data traversal or query reuse. You choose the change; do not assume an example must help.

Compare your first parallel version with the optimized version at **the same four-thread configuration**, workload, compiler policy and timing boundaries, three repetitions each. Reuse the already collected before timings. Keep all correctness checks. Preserve both versions and show the changed code in the PDF. Report a slowdown honestly; explain what your evidence supports.

The full-sort baseline, a parallel implementation and a changed selection algorithm answer different performance questions. Label **parallel speedup** separately from **overall improvement over the supplied baseline**; do not call an algorithm change pure strong scaling.

### C. Sensitivity

Change **only Q from 32 to 64**, keeping N=65536, D=128, K=10 and four threads. Rerun the before/after optimization versions three times each. Obtain a serial reference for Q=64 for correctness. Compare search-phase throughput as well as elapsed time. Do not mix Q=64 into the Q=32 fixed-workload speedup calculation.

### D. Hybrid design and available resources

Where the established SciTech MPI toolchain is available, implement collection sharding and a correct global top-k merge. Test **2 MPI ranks × 2 OpenMP threads** against **1 rank × 4 threads** for the same workload and implementation algorithm, three repetitions. Use a bounded single-node allocation. Request appropriate MPI thread support; MPI calls outside OpenMP regions can use FUNNELED. Record actual rank/thread counts, placement and the compatible launcher.

If the site MPI toolchain/launcher is unavailable, show the attempted environment/launch diagnostic and use the **CPU alternative**: compare two correct OpenMP decompositions (for example, work over queries versus collection rows) at four threads, three repetitions each, with equivalent algorithm and work. Include the proposed MPI shard ownership and global merge algorithm in your report. This documented alternative can earn the same marks; an unsupported assertion that MPI is unavailable cannot replace evidence. A single-node MPI result is not proof of multi-node scaling.

### E. Reproducibility and portability

Record modules, compiler/version, flags, source hash, input generator/parameters, commands, job IDs, host/CPU information, Slurm allocation and thread settings. Write an exact small rerun procedure.

If a trusted C/OpenMP-compatible Apptainer image already exists and is accessible, compile and run the small correctness case inside it through Slurm, check against the reference and record its checksum. Do not use an unverified image path or download/install an image for this task. If no compatible image exists, provide the availability/compatibility diagnostic and a fully specified module-based rerun recipe; explain the remaining portability limits. Either evidenced route can earn full reproducibility marks. GPU execution and containerized MPI are not required.

## 5. Resource and measurement discipline

- Start with the small correctness cases. Use at most **one node, four CPUs total, 1 GB, six minutes per job** for the default experiments. Keep jobs sequential where practical; do not submit a large sweep. Do not add an account or QoS.
- `ranks × threads` must fit the CPU allocation. Prevent hidden nested threading and record binding settings. Confirm the CPU partition with `sinfo`.
- Time the complete batch of Q searches, including scoring, selection and any required global merging/synchronization. For MPI use a global maximum elapsed time across ranks. Exclude generation, normalization, compilation and scheduler queue wait consistently. State separately that you are not measuring full service latency.
- Retain all repetitions, not only the fastest. If timings are too short/noisy, report that limitation. Any instructor-approved workload adjustment must apply to all compared configurations and be declared; do not silently change N or Q.
- PASS means a completed numerical check passed. FAIL means an attempted failure. SKIPPED means an optional/unavailable feature. NOT RUN means no execution evidence. A slower correct run is not a numerical failure.
- If the cluster is blocked, retain diagnostics, identify affected experiments as NOT RUN and contact the instructor before the deadline. Local runs may help debugging but do not fulfil the required cluster evidence by themselves. Never fabricate missing results.

## 6. Blackboard submission: two graded parts

### Part A · Eight evidence questions — 40 points

Answer each question in its Blackboard answer field. Use your own run values, job IDs and PDF page/table references. Usually 80–120 words per question is sufficient. These are manually graded assignment questions, **not the separate module quiz**.

**Q1 — Work and memory (5 points).** For the main workload, calculate the collection storage in bytes/MiB and the number of query–collection dot products and component multiply-add pairs. Explain what changes when Q doubles and which costs your timer excludes. Distinguish a multiply-add pair from a FLOP count.

**Q2 — Parallel ownership (5 points).** Explain your actual OpenMP ownership and race-prevention design. For a four-CPU budget compare 1×4 and 2×2, including the global top-k merge. Cite your executed hybrid/CPU-alternative evidence and distinguish executed work from a proposed design.

**Q3 — Correctness under ties (5 points).** Give the ordered expected IDs for the supplied tie fixture, show your actual result and checker outcome, and explain why checking only the first neighbour or a checksum is insufficient.

**Q4 — A defensible speedup (5 points).** Using your three-repeat measurements, calculate the median, speedup and efficiency for your first OpenMP implementation. State the reference denominator, identify whether you are reporting strong scaling or overall baseline improvement, and explain why Q=64 cannot be substituted into the Q=32 comparison.

**Q5 — Optimization decision (5 points).** State the measured bottleneck hypothesis, the single implementation change, correctness outcome and same-configuration before/after result. Explain whether the evidence supports retaining the change, even if it slowed down.

**Q6 — Interpreting the limit (5 points).** Use phase timings or profiler evidence and repetition variation to explain one observed performance limit. Separate a measured observation from an untested explanation, and propose one further controlled measurement that could distinguish competing explanations.

**Q7 — Reproducing the result (5 points).** Give the essential environment and exact rerun information, cite your container result or documented module-based alternative, and explain two relevant dependencies that a container image alone would not settle.

**Q8 — System recommendation (5 points).** Recommend a deployment configuration for this measured workload, considering runtime, search-phase throughput, CPU budget and reproducibility. Connect your recommendation to at least two earlier-course concepts, such as memory behaviour, Amdahl's law, communication, resource placement or data movement. State one limit on extrapolating your findings to a larger system.

### Part B · One consolidated evidence PDF — 60 points

Upload **one PDF**, named `M5S5_StudentID_VectorSearch.pdf`. It must contain everything needed to review your work without opening a notebook, repository or separate ZIP. A repository link is optional supplementary access, not a replacement for the PDF.

Use **4–6 pages for the main report**, including figures, followed by readable technical appendices as needed:

1. Identification, source/starter commit, application and timing definition.
2. Architecture and decomposition, including actual allocations and the hybrid/alternative route.
3. Correctness table for all required fixtures and configurations, with representative ordered top-k output and checker outcomes.
4. Experiment table with all raw repetition times, medians, speedup/efficiency where valid, throughput, source version and job IDs. Include Q=64 separately.
5. Bottleneck evidence, optimization comparison, interpretation and limitations.
6. Reproduction instructions, environment and image identity/alternative evidence.

Append **the complete source of your submitted candidate and any separate before-optimization version needed to reproduce the comparison**, your build/Slurm/analysis scripts, exact commands and a concise log extract connecting each results set to its job and configuration. The baseline may be referenced by its provided file hash/commit; identify any modifications explicitly. Include your raw timing table as text, not only a graph. Keep complete logs and full result JSON in your own working area in case verification is requested; include the relevant numerical evidence and checks in the PDF rather than dumping every repeated neighbour list.

Use selectable, legible code text, page numbers and clear labels. No screenshot-only code, external-only results or unexplained graphs. State any permitted assistance and sources under the course rules. You must be able to explain and reproduce your work. The guided notebook does not grant permission to submit AI-generated final assessed code or answers where the course rules prohibit it.

## 7. Internal marking

| Component | Points |
|---|---:|
| Part A: eight questions, 5 points each | 40 |
| PDF: correctness and top-k evidence | 15 |
| PDF: controlled methodology and complete timing evidence | 10 |
| PDF: cluster execution and hybrid/CPU alternative | 10 |
| PDF: bottleneck investigation and optimization comparison | 10 |
| PDF: reproducibility and portability evidence | 10 |
| PDF: clarity, self-contained code and traceable presentation | 5 |
| **Total** | **100** |

Part A assesses your reasoning; Part B assesses the implementation and auditable evidence. A correct negative performance result can earn full marks. Fast timings with failed correctness checks cannot support a validated improvement claim.
