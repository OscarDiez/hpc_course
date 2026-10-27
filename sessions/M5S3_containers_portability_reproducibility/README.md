# M5.S3 Modules and Apptainer

Open [M5S3_containers_portability_reproducibility.ipynb](M5S3_containers_portability_reproducibility.ipynb).

Each step explains the purpose, shows SSH commands and executes those exact commands in the matching notebook cell. Run All follows the practical in order and prints command output and waiting messages. No GPU, package installation, image download or build is needed.

The student route is:

1. Inspect modules, paths and the current environment.
2. Select foss/2023b, compile a readable C program immediately and run it on an allocated core. Check its known sum and compiler identity.
3. Save, purge and restore a named module collection; compare compiler paths.
4. Inspect an HDF5 module and its compiler wrappers.
5. Inspect Apptainer and the existing shared PyTorch image on an allocated CPU node.
6. Train the same small model in separate container processes; compare A and B.
7. Change only the target slope and check run C.
8. Use an explicit /work bind, repeat the same code and verify a file written from the container on the host.
9. Read the final status/model tables and plan a one-parameter experiment.

The notebook's Python interpreter does not need PyTorch. The model uses the interpreter inside the existing course image at `/data/software/containers/pytorch-2.8.0-cuda12.8.sif`. The path and image interpreter are visible in `container.env`. Every substantive calculation uses an explicit Slurm CPU allocation. The shell helper removes inherited Jupyter allocation limits from child requests, preserving Slurm configuration/account/QOS settings.

Each cell starts a separate shell. Compiler cells explicitly load their module stack. Module selection does not change the existing notebook interpreter. The module collection exercise runs save/purge/restore in the same shell and stores a collection called m5s3_class in the student's account.

The three readable source/configuration files are in [basic_lab](basic_lab) and embedded as writefile cells. There is no automatic image-discovery framework or large generated worker script in this route. Missing runtime/image is SKIPPED; an attempted incompatible execution is FAILED. Dependent training/bind stages require a checked baseline, and the final table does not turn skipped work into a pass.

The [extended notebook](M5S3_containers_portability_reproducibility_extended.ipynb) preserves the previous complete notebook, including dependency drift, temperature calculations, virtual environments, NumPy, caching and handover. See [its earlier guide](README_extended.md). Existing examples and historical validation records remain available. `validate_notebook.py` now points at that extended notebook.

Local checks for the new route are documented in [VALIDATION_basics.md](VALIDATION_basics.md). They use actual C/PyTorch payloads with explicit CLI test doubles; actual SciTech and Apptainer/SIF execution still require confirmation.

Raw notebook URL:

https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s3-reproducibility-2026/sessions/M5S3_containers_portability_reproducibility/M5S3_containers_portability_reproducibility.ipynb
