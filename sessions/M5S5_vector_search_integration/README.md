# M5S5 · Vector search integration lab

Guided C lab consolidating hybrid computing, MPI/OpenMP layout, performance evidence and reproducibility. This supports Session 25; the assessed group milestone remains a separate submission. Module quiz and individual cluster assignment are separate activities.

## Open the notebook

[Notebook](M5S5_vector_search_integration.ipynb)

JupyterLab **File → Open from URL**:

https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s5-vector-search-2026/sessions/M5S5_vector_search_integration/M5S5_vector_search_integration.ipynb

Save your own copy. Run the Python 3 cells in order. The notebook embeds every source/helper/batch file, so opening the raw notebook alone is sufficient. It creates a new owned working folder; no packages are installed. Read and edit `site_env.sh` using the actual SciTech module names before submitting.

## Five experiments

1. Independent tiny Python reference versus actual serial C top-1 search.
2. Serial / OpenMP 1, 2, 4 threads, three repetitions each in a separate Slurm job.
3. Optional hybrid 2 MPI ranks × 2 OpenMP threads, same four-CPU budget.
4. Optional existing compatible Apptainer C/OpenMP image; compile inside it and check numerical results.
5. Actual-record tables/plot, phase evidence and a one-parameter query-count experiment.

The generated data represent a retrieval workload but are not real semantic embeddings. Exact top-1 IDs and scores are checked, with smaller ID breaking exact score ties. Generation and normalization are outside the search timing region. Search-phase throughput is not service latency.

## Cluster terminal route

Use the [SciTech guide](../../cluster/README.md). If using the repository files directly, copy this session directory to an owned work area first. In that directory:

```bash
# Edit site_env.sh to load the site's actual GCC/OpenMP toolchain.
sbatch cpu.sbatch
squeue -u "$USER"
# Inspect m5s5-cpu-JOBID.out and results/*.json.
# Optional, after loading matching MPI modules:
sbatch hybrid.sbatch
# Optional, with an existing trusted gcc/OpenMP image configured in site_env.sh:
sbatch container.sbatch
# Sensitivity: change only the query count, keep other settings fixed:
sbatch --export=ALL,M5S5_QUERIES=8 cpu.sbatch
```

No account/QoS, downloads, direct compute-node SSH, GPU allocation or package installation is required. Each job requests one node, four CPUs total, 1 GB, six minutes. Do not submit all optional experiments unless useful. Queue wait is additional. The notebook does not block waiting indefinitely or resubmit automatically.

## Evidence and validation

PASS is an executed numerical check; FAIL an attempted error; SKIPPED an unavailable/disabled optional route; NOT RUN has no execution evidence. Job submission alone is not PASS. If the job fails before the runner starts, inspect the scheduler/log; absence of JSON does not prove success.

Instructor local validation: all seven notebook code cells executed; tiny independent Python/C reference passed; serial and OpenMP passed for 1/2/4 threads, including odd/small sizes; twelve default-workload repetitions checked against the serial reference; deliberate wrong-result rejection passed; batch files passed shell syntax checks. No optional MPI, Slurm or Apptainer execution occurred locally. **SciTech validation is NOT RUN.** Module names, launch integration and compatible image path must be verified on site. Cleared notebook outputs contain no fabricated performance results. Optional plotting uses an existing matplotlib installation; the table/CSV work without it.

Primary reference: official 2026–27 syllabus Sessions 21–25. Repository conventions: M4S5 guided lab, M5S3 container notebook, cluster guide. No change to course grading weights is made by this notebook.
