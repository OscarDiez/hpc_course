# M5.S2 — From basic MPI/OpenMP to scalable applications (2026–27)

**Official course:** IE University, Bachelor in Computer Science and Artificial Intelligence, HPC (2026–27). **Session 22, pages 15–16** of the syllabus supplied by the lecturer. The emphasis is on **application design**, not learning advanced MPI APIs.

## Open in JupyterLab
1. Open https://jupyter.iesci.tech:8085/ with your IE login.
2. **File → Open from URL…**
3. Paste the current raw GitHub URL:
```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s2-advanced-mpi-notebook-2026/sessions/M5S2_scalable_mpi_openmp/M5S2_scalable_mpi_openmp.ipynb
```
4. Choose Python 3 and save a personal copy.

[Notebook on GitHub](https://github.com/OscarDiez/hpc_course/blob/teaching/m5s2-advanced-mpi-notebook-2026/sessions/M5S2_scalable_mpi_openmp/M5S2_scalable_mpi_openmp.ipynb)

**After review and merge**, use the corresponding URL with `main` instead of the temporary teaching branch.

## Teaching scope and alignment
- **Previous (Session 21):** Hybrid computing patterns: MPI between nodes, OpenMP within nodes and optional GPU offload.
- **This session (Session 22):** From basic MPI/OpenMP to scalable applications: combine ranks, threads, synchronization, decomposition and bottleneck reasoning.
- **Next (Session 23):** Containers, portability and reproducibility.

Required concepts: what to distribute across processes, what to parallelize within a process, what data to communicate, when to synchronize, and how to interpret scaling limitations. Advanced MPI concepts are **optional background** only, as stated in the official 2026–27 syllabus.

Main reading: Oscar Diez, *Foundations of High-Performance Computing*, Chapters 7 and 8.

## Notebook and 80-minute class plan

The notebook first runs two real small **local Python** checks, then creates **one C application** implementing a heat-diffusion Jacobi stencil with MPI rank decomposition, halo exchanges and an OpenMP parallel loop. The same code runs in four configurations; a smaller grid is added to discuss overhead.

| Time | Topic | Notebook section |
|---|---|---|
| 0–10 min | Open JupyterLab, predict scientific outcome | 0–1 |
| 10–25 min | Decompose the plate and identify rank/thread boundaries | 2–3 |
| 25–35 min | Prepare a safe Slurm allocation and predict outcomes | 4–6 |
| 35–45 min | MPI ranks and halo messages | 7 |
| 45–55 min | OpenMP threads and shared data | 8 |
| 55–65 min | Compare 4×1 MPI/OpenMP with 2×2 hybrid | 9 |
| 65–80 min | Overhead, correctness, measurements, checkpoints | 10–12 |

**Student one-parameter activity:** Change only `GRID_N` (48 / 96 / 192) or `STEPS` (20 / 40 / 80). Re-run the preparation and submission cells. Observe numerical correctness before interpreting speed.

## SciTech cluster safety
Read [cluster/README.md](../../cluster/README.md) and inspect `sinfo` and `module avail` first. Set `RUN_CLUSTER=True` only after confirming access and the `cpu` partition / intended MPI module. Default `RUN_CLUSTER=False` does **not** submit a job. No MPI work should be run directly on login nodes.

Slurm job: one CPU node, nominally **4 CPUs allocated**, **1 GiB**, **6 minutes**. The same executable is compiled with `mpicc -fopenmp`, and five sequential `srun` steps vary process/thread organization. The job may need cluster-specific Slurm binding adjustments: record error messages and keep the affected step **FAIL / NOT RUN** until confirmed; do not silently oversubscribe or request excessive resources. No GPU or package installation is necessary.

For each run the C program builds a **serial reference** for the same grid/steps, checks maximum absolute numerical error and prints `RESULT status=PASS/FAIL`. The `model_halo_payload_bytes` is a theoretical data-transfer count, not measured network traffic. Timing differences from one short single-node run are not proofs of large-scale scalability.

## Validation status
- **Verified locally:** independent Python reference and balanced row decomposition on a 12×12 / 14-row test case.
- **Static review:** notebook JSON, program generation, MPI/OpenMP symbols, Slurm defaults, and PASS/FAIL/NOT RUN output contract.
- **NOT RUN on SciTech:** MPI+OpenMP compilation, allocation, CPU binding, correctness comparisons, elapsed times and module support. Before the notebook is student-ready, obtain an actual short SciTech Slurm output file and confirm **S/M/O/H/B** PASS or document genuine exceptions.
- **Do not merge this draft PR until a real runtime run has passed or the limitations are explicitly accepted.**

The **old Advanced MPI notebook created earlier in this draft PR was based on an outdated syllabus and is superseded**. It must not be used for the 2026–27 lecture. The 2026–27 student AI policy applies to assessed submissions; providing teacher-authored demonstration code does not authorize AI-generated assessed work.


## New slide-linked mini demos (43-cell enhanced notebook)

The main heat-diffusion solver remains the **same single coherent scientific application**. Students now have small, independent demos for concepts that would otherwise remain abstract.

| Demo | Close connection to lecture | Real code | Expected observation | Evidence source |
|---|---|---|---|---|
| A. Heat spreading | Stencil update, old/new data dependence | Python | Hotspot spreads; outer boundary stays fixed | LOCAL computation and optional LOCAL plot |
| B. Row ownership/halo | MPI data decomposition and neighbor messages | Python | Rank ownership map and exact halo partners | LOCAL logical model (not MPI execution) |
| C. Four core layouts | MPI-only, OpenMP-only, MPI+OpenMP | Python | 1×1, 4×1, 1×4 and 2×2 worker diagrams | LOCAL configuration illustration |
| D. Lost update | Race condition, synchronization, reduction | Python | Deliberate interleaving produces 1 instead of 2 | LOCAL deterministic teaching illustration, **not** a real data race |
| E. MPI ranks + collective | Rank ID, `MPI_Sendrecv`, `MPI_Allreduce` | **Real C/MPI** | Four ranks contribute 1+2+3+4 = 10; each observes a neighbor's value | Actual `srun` log required |
| F. OpenMP workers + reduction | `parallel`, `for`, `reduction` | **Real C/OpenMP** | Two threads share eight iterations; sum = 36 | Actual `srun` log required |
| G. Cost and overhead | Why strong scaling can fail | Python | Illustrative computed values for 1, 2 and 4 ranks | **MODEL**, arbitrary units, never measured seconds |
| H. Five MCQs | Misconceptions and decision-making | Markdown | Explain each answer before revealing | Student discussion, not code execution |

**Real C mini programs E/F are compiled and executed as two additional small steps inside the same existing bounded Slurm allocation.** This is not another submitted job. The notebook checks `MINI_RESULT` **and** matching `MINI_END` before it can label these demos PASS.

### Teaching selection (80-minute target)
Start with A and B for the problem and decomposition, demonstrate C and D briefly, then use E/F to show rank/thread identities, and run the original heat-solver cases S/M/O/H/B as time permits. Close with one bottleneck comparison and the MCQs. The cost-model G and detailed timing explorations are optional self-study. Avoid attempting every extension in the live session.

### Local validation performed during authoring
- Python reference heat solver, three snapshots and fixed boundaries: **PASS** (local).
- Four-rank ownership covers all interior rows and identifies boundaries: **PASS** (local).
- Deliberate lost-update sequence and safe mathematical sum: **PASS** (local).
- Cost model internal consistency: **PASS** (local; it is not physical performance evidence).
- OpenMP C mini program: **COMPILED and PASS locally with GCC -fopenmp**; exactly two threads and numerical sum 36. This is **not a SciTech run**.
- Uploaded HTML snapshot from an IE environment showed `mpicc`, `sbatch` and `srun` commands visible, and Python reference computation completed. That snapshot kept `RUN_CLUSTER=False`, so **no actual MPI/Slurm execution evidence** was present in it.
- **Still NOT RUN:** MPI mini program, MPI+OpenMP heat solver modes, actual SciTech module/partition and Slurm binding. Do not merge before runtime acceptance.

**Performance caveat:** First-time OpenMP startup and very small grid sizes can dominate timings. The notebook does not promise speedup or extrapolate a single-node classroom result to multiple nodes.
