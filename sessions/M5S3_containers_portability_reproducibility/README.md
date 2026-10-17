# M5S3 · Containers, portability and reproducibility in HPC

Syllabus Session 23, 2026–27. This session follows scalable MPI + OpenMP.

- [Open the notebook](M5S3_containers_portability_reproducibility.ipynb)
- [Raw notebook for JupyterLab](https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M5S3_containers_portability_reproducibility/M5S3_containers_portability_reproducibility.ipynb)

Use **File → Open from URL** in JupyterLab and paste the raw link. Run cells in order with Python 3.9+.

The core examples require only the standard library. They demonstrate dependency changes, process environments, a real empty virtual environment, a Spack specification, a recorded temperature analysis, floating-point accumulation, cache invalidation and a reproducibility handover. Twelve MCQs provide feedback after students enter a letter.

The notebook includes real outputs from its default CPU path. Apptainer, MPI, GPU and Slurm integration are optional cluster exercises. No job submission, image download or package installation happens automatically. Default validation did not execute a container or a multi-node/GPU job.

## Class timing (80 minutes)

| Minutes | Notebook sections |
|---|---|
| 0–15 | 0–1: environment and dependencies |
| 15–25 | 2–3: modules, virtual environments and Spack |
| 25–40 | 4 and 7: container boundaries and commands |
| 40–55 | 5–6: recorded experiment and numerical agreement |
| 55–65 | 8: Slurm, MPI and GPU integration |
| 65–80 | 9–10: workflow cache and handover |

Students can run core sections in numerical order and return to the optional cluster cells later. The older `chapters_examples/23.M5.S3.Hybrid_Computing(5_3).ipynb` remains a legacy hybrid-programming example and has not been replaced.
