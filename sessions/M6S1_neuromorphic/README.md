# M6S1 — Neuromorphic computing

Use this **one-session notebook**, following the revised teaching scope. About 30–40 minutes, with optional external practice. Other Module 6 subjects belong in separate session notebooks.

- [Notebook](M6S1_neuromorphic.ipynb)
- [Raw notebook — paste into JupyterLab File → Open from URL](https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m6s1-alternative-paradigms-2026/sessions/M6S1_neuromorphic/M6S1_neuromorphic.ipynb)
- [Validation](VALIDATION.md)

## What students do

1. Read a short C neuron model and change its firing threshold.
2. Reuse that code to explore leakage and retained state.
3. Read a short sensor loop and compare samples with change messages.

Each exercise includes a problem, prediction, visible C code, execution, output, explanation, one-parameter/input change and takeaway. No ML packages or specialised hardware. Python only handles files/compilation/execution. Tables show real saved local outputs. The optional external task links to snnTorch's editable Colab tutorial and EPFL exercises with specific before/after observations to collect.

## Run

Save a personal copy, select Python 3, and run top to bottom. GCC is required; use the existing [cluster guide](../../cluster/README.md) to load the course compiler if needed. Files are isolated in `~/hpc_m6s1_neuromorphic/run_*/`. The runner uses a one-task Slurm `srun` on systems exposing Slurm and otherwise runs the tiny examples locally. SciTech requests one CPU, 1 GB, two minutes in `cpu`; no MPI/GPU or package installation. Slurm may queue; execution errors are labelled FAIL. GCC absence reports SKIPPED. Before execution results are NOT RUN. PASS applies to completed output checks, not hardware suitability.

## Teaching scope

This replaces the earlier broad alternative-paradigms notebook **as the student entry point for M6S1**. The old notebook and its 29-slide plan are retained as supplementary source material; that plan covers a broader scope and is not the plan for this neuromorphic-only lesson. No PowerPoint has been modified.

Saved outputs: LOCAL_CPU. Actual SciTech and JupyterLab URL-import validation: NOT RUN.
