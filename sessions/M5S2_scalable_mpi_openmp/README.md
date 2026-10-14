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
