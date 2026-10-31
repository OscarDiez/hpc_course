# M6S1 · Quantum, neuromorphic and alternative computing paradigms

Syllabus Session 26, BCSAI, IE University, 2026–27. This follows Module 5 reproducible workflows and leads to Session 27 exascale/post-exascale computing.

- [Notebook](M6S1_quantum_neuromorphic_alternative.ipynb)
- [Raw notebook for JupyterLab](https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m6s1-alternative-paradigms-2026/sessions/M6S1_quantum_neuromorphic_alternative/M6S1_quantum_neuromorphic_alternative.ipynb)
- [C and execution files](../../session_demos/26_alternative_paradigms/)
- [Validation report](VALIDATION.md)
- [Complete 29-slide restructuring plan and implementation checklist](SLIDE_PLAN.md)

Open the raw link with **File → Open from URL** in JupyterLab. Save a personal copy. Run cells in order with Python 3.9+. No packages, images, accounts or datasets are downloaded. The source files are embedded, so a repository checkout is unnecessary. Matplotlib is optional; printed tables work without it.

| Syllabus outcome | Notebook experiment | Evidence |
|---|---|---|
| Explain neuromorphic event processing | E1 · Sparse sensor updates | Executed C CPU simulation plus operation counts, no energy claim |
| Understand neuron state and spikes | E2 · A neuron remembers recent input | Executed discrete C simulation, no training or neuromorphic hardware |
| Explain in-memory and photonic computation | E3 · One matrix operation, several hardware mappings | Executed C arithmetic and quantization error model, no device simulation |
| Explain quantum fundamentals and limits | E4 · Quantum interference before measurement | Exact one-qubit C CPU simulation and seeded CPU shot sampling |
| Assess complementary hybrid workflows | E5 · The complete hybrid workflow | Explicitly illustrative cost model, no accelerator measurement |

Every experiment follows real problem, prediction, code, run, result, explanation, one-parameter change and takeaway. Five MCQs provide feedback after students choose A–D.

## SciTech

Use the [cluster access guide](../../cluster/README.md) and [infrastructure notes](../../cluster/COURSE_VALIDATION_NOTES.md). The default route uses Slurm if `sbatch` is visible. It submits a **separate CPU allocation: one node, one task, one CPU, 1 GB, two-minute limit**. It clears inherited conflicting memory variables before submission. The job loads `foss/2023b` when the module function exists and otherwise checks the inherited GCC. Missing GCC reports SKIPPED; an attempted failed compilation reports FAIL.

Submission and collection are separate: rerun the collection cell when the job finishes. The run folder and saved job ID avoid duplicate submission of identical settings across restarts. Source/settings fingerprints isolate results. Change one setting and rerun from Settings. Set `REPEAT=True` only for an intentional repeat after the previous job has left the queue. Inspect `squeue`, `sacct` and the exact job log if results are absent. A pending or absent result is NOT RUN.

Outside Slurm, the notebook executes only tiny concept/correctness examples. None are performance benchmarks. No specialized hardware is required. MPI, GPU, QPU, neuromorphic, photonic and resistive-device execution are explicitly SKIPPED. Optional OpenMP checks numerical agreement, without timing claims. To request it, set `openmp=True` and `CPUS=2`.

## Teaching route

The 80-minute lecture uses selected notebook cells and short predictions. Students complete parameter changes afterwards. E1/E2 support the neuromorphic block, E3 links in-memory and photonic arithmetic, E4 supports the quantum block, and E5 returns to application-level HPC choices. Do not devote live class time to the embedded orchestration source.

## Validation boundary

All 17 notebook code cells executed locally in order with Python 3.12 and GCC 13.3.0. C compiled with warnings as errors. Core E1–E5 passed, including independent Python reference checks. Optional two-thread OpenMP passed in a separate local test. See `validation_local_results.json` and `validation_path_checks.txt`.

Notebook validation checked JSON/cell structure, parsed Python syntax and executed code cells with captured output. It did not launch a Jupyter frontend or test JupyterLab's Open from URL UI. SciTech Slurm scheduling, module exposure and compute-node execution are **NOT RUN**. Mocked scheduler tests are not cluster measurements. Saved notebook outputs are visibly labelled LOCAL_CPU.
