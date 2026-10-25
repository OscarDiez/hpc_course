# M5S5 · Individual cluster assignment
## Batched top-k vector search: parallelize, diagnose and justify

**Individual work.** This is separate from the guided notebook, the Module 5 quiz and the group project milestone. Submit your own cluster evidence and explanations. Deadline and course-grade weight are those announced in Blackboard; the 100 points below describe this assignment's internal marking only.

## 1. The application

A retrieval service receives a batch of query vectors. For each query, it must return the K collection items with the largest cosine similarity. This is the numerical search operation used in recommendation and retrieval pipelines. The supplied deterministic vectors stand in for embeddings: you are evaluating computational behaviour, not semantic relevance or a language model.

You receive a correct, deliberately simple **serial C baseline**. It stores normalized float vectors, accumulates dot products in double precision, sorts every collection item for each query, and returns the ordered top-k. You must develop your own parallel version, measure it through Slurm, investigate a bottleneck and implement one further optimization. A speedup is not guaranteed and is not itself the grading target.

This extends the guided top-1 example: a correct local winner is insufficient for a complete, ordered top-k. Merely changing the guided program's name or repeating its timings does not complete this task.

## 2. Starter files

Use `topk_baseline.c`, `check_topk.py`, `baseline.sbatch`, `site_env.sh` and `container_replay.py` from this assignment directory. Preserve the original baseline. Put your implementation in `topk_candidate.c` and record the source hashes and starter repository commit.

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

### E. Environment modules, Apptainer and a reproducible workflow

This is a **required hands-on activity**, not just a paragraph about containers. The teacher will identify an existing trusted Python 3 SIF on SciTech. You do not need gcc, MPI or a package installation inside this image.

1. Inspect and record the native environment: `module list`, `type -a gcc mpicc python3 apptainer`, compiler version and `mpicc --showme` where available. Put the chosen site modules in `site_env.sh`. Select a matching compiler/MPI stack; do not mix a system MPI wrapper with an unrelated EESSI toolchain. Record the running notebook interpreter separately from shell module changes.
2. Run your actual C searches through Slurm and retain the serial reference and candidate JSON. C computation remains native; the following container stage validates and analyses its recorded results.
3. Run the supplied **unchanged** `container_replay.py` with native Python, then with Apptainer using the same reference/candidate files. Its `--reference`, `--candidate` and `--output` arguments name the exact files. Use your own candidate's main workload (Q=32, K=10), not a copied guided top-1 result.
4. Bind the input/script directory read-only and a separate owned output directory read-write. Use `--cleanenv`. Compare numerical results and code/input hashes; interpreter/platform metadata can differ. Record the SIF checksum, exact launch, host and job ID. A matching checksum identifies bytes, not publisher trust.
5. Set `HPC_M5S5_HOST_SENTINEL=host-only` in the native environment and inspect its recorded value inside the clean container. Then explicitly forward a chosen value with `APPTAINERENV_HPC_M5S5_HOST_SENTINEL` and verify it is recorded without changing scientific results. Explain what this shows about inherited versus explicitly supplied environment settings.
6. Copy one candidate result, deliberately alter a score, and show that validation inside the container fails. Preserve the original. This is an expected rejection test, not permission to submit changed benchmark results.
7. Write a short executable shell workflow that checks its inputs, runs the native and container validation stages, stops on unexpected errors and saves separate outputs/logs. Include commands for one small independent rerun and for rebuilding/rerunning after a source change. Show why an old PASS report cannot validate newly edited code or different inputs.

Example launch pattern, **inside your Slurm allocation**, with directory/image variables set to actual absolute paths:

```bash
srun -n1 -c1 apptainer exec --cleanenv \
  --bind "$INPUT_DIR:/work:ro" --bind "$OUTPUT_DIR:/out" \
  --pwd /work "$M5S5_IMAGE" python3 /work/container_replay.py \
  --reference /work/reference.json --candidate /work/candidate.json \
  --output /out/container_check.json
```

Prepare the chosen files in `INPUT_DIR` and create the owned `OUTPUT_DIR` before launching. `M5S5_IMAGE` must point to the existing approved Python 3 image; it is not an image download command. If Apptainer/image access is blocked, retain the diagnostic and report this task as **NOT RUN**; do not silently replace required execution with a theoretical answer. The instructor will resolve the site issue or state the applicable alternative before release.

This validates **native C outputs inside a container**. It does not demonstrate C/MPI/GPU execution inside that image. A C-in-container extension is optional and requires a separately compatible toolchain/runtime. GPU execution, Spack installations and advanced workflow-framework syntax are not required. In your explanation distinguish what modules, a package specification, an image and a workflow preserve.

## 5. Resource and measurement discipline

- Start with the small correctness cases. Use at most **one node, four CPUs total, 1 GB, six minutes per job** for the default experiments. Keep jobs sequential where practical; do not submit a large sweep. Do not add an account or QoS.
- `ranks × threads` must fit the CPU allocation. Prevent hidden nested threading and record binding settings. Confirm the CPU partition with `sinfo`.
- Time the complete batch of Q searches, including scoring, selection and any required global merging/synchronization. For MPI use a global maximum elapsed time across ranks. Exclude generation, normalization, compilation and scheduler queue wait consistently. State separately that you are not measuring full service latency.
- Retain all repetitions, not only the fastest. If timings are too short/noisy, report that limitation. Any instructor-approved workload adjustment must apply to all compared configurations and be declared; do not silently change N or Q.
- PASS means a completed numerical check passed. FAIL means an attempted failure. SKIPPED means an optional/unavailable feature. NOT RUN means no execution evidence. A slower correct run is not a numerical failure.
- If the cluster is blocked, retain diagnostics, identify affected experiments as NOT RUN and contact the instructor before the deadline. Local runs may help debugging but do not fulfil the required cluster evidence by themselves. Never fabricate missing results.

## 6. Your own diagnostic experiment

Choose one additional small test motivated by your actual results. Before running, record the observation, hypothesis, single setting/detail you will change and predicted outcome. Execute it within the resource limits and explain whether the evidence supports your prediction. Preserve its job ID, configuration and raw output. Repeating the guided notebook unchanged is not an independent diagnostic experiment.

## 7. Blackboard submission: two graded parts

### Part A · Assignment question component — 40 internal points

Complete the questions selected from the assignment evidence bank in Blackboard. They concern these files, correctness checks, configurations, code behaviour, measurements and workflow decisions. Keep your own working directory and PDF references available. This is separate from the Module 5 quiz. The pool's raw marks are normalized/configured by the instructor to this component's allocation; they do not define a new course-grade weight.

### Part B · One consolidated evidence PDF — 60 points

Upload **one PDF**, named `M5S5_StudentID_VectorSearch.pdf`. It must contain everything needed to review your work without opening a notebook, repository or separate ZIP. A repository link is optional supplementary access, not a replacement for the PDF.

Use **4–6 pages for the main report**, including figures, followed by readable technical appendices as needed:

1. Identification, source/starter commit, application and timing definition.
2. Architecture and decomposition, including actual allocations and the hybrid/alternative route.
3. Correctness table for all required fixtures and configurations, with representative ordered top-k output and checker outcomes.
4. Experiment table with all raw repetition times, medians, speedup/efficiency where valid, throughput, source version and job IDs. Include Q=64 separately.
5. Bottleneck evidence, optimization comparison, interpretation and limitations.
6. Reproduction instructions, environment and actual image identity, native/container comparisons, environment forwarding and expected-error evidence.
7. Your independent diagnostic experiment: prediction, controlled change, result and interpretation.

Append **the complete source of your submitted candidate and any separate before-optimization version needed to reproduce the comparison**, your build/Slurm/analysis scripts, exact commands and a concise log extract connecting each results set to its job and configuration. The baseline may be referenced by its provided file hash/commit; identify any modifications explicitly. Include your raw timing table as text, not only a graph. Keep complete logs and full result JSON in your own working area in case verification is requested; include the relevant numerical evidence and checks in the PDF rather than dumping every repeated neighbour list.

Use selectable, legible code text, page numbers and clear labels. No screenshot-only code, external-only results or unexplained graphs. State any permitted assistance and sources under the course rules. You must be able to explain and reproduce your work. The guided notebook does not grant permission to submit AI-generated final assessed code or answers where the course rules prohibit it.

## 8. Internal marking

| Component | Points |
|---|---:|
| Part A: assignment evidence questions | 40 |
| PDF: correctness and top-k evidence | 15 |
| PDF: controlled methodology and complete timing evidence | 10 |
| PDF: cluster execution and hybrid/CPU alternative | 10 |
| PDF: bottleneck investigation and optimization comparison | 10 |
| PDF: reproducibility and portability evidence | 10 |
| PDF: clarity, self-contained code and traceable presentation | 5 |
| **Total** | **100** |

Part A assesses your reasoning; Part B assesses the implementation and auditable evidence. A correct negative performance result can earn full marks. Fast timings with failed correctness checks cannot support a validated improvement claim.
