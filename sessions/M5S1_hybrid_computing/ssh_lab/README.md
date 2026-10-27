# M5S1 SSH heat laboratory

Open `../M5S1_hybrid_computing.ipynb` in SciTech JupyterHub and choose Run All. The default route creates files, runs CPU solvers/libraries, tries a ParaView module, runs GPU heat solvers, collects a CPU profile, and validates/displays results. Nine checkpoints support the manual SSH route.

Matplotlib is not required. Notebook previews use standard-library SVGs. ParaView uses its own `pvpython` interpreter and module in a separate CPU allocation. It exports VTI, centre-line CSV, PNG and PVSM. A missing installation reports SKIPPED; failed rendering reports FAIL.

The settings at the beginning control problem size, repetitions, optional stages, queue waiting and the ParaView module version. Module lists, compiler/wrapper information, linked libraries, source/binary hashes and Slurm allocation details are recorded. MPI launcher selection uses a tiny numerical preflight; `auto` tries srun then mpirun and records the validated choice.

For manual SSH work, run from this directory on the login node:

- CPU: `N=512 STEPS=500 REPEATS=3 bash scripts/submit.sh cpu all`
- GPU: `N=512 STEPS=500 REPEATS=3 bash scripts/submit.sh gpu all`
- ParaView: `HEAT_INPUT=/absolute/path/reference-r0.bin N=512 bash scripts/submit.sh paraview all`

Use a serial, openmp, mpi, hybrid or libraries CPU stage instead of all for individual exercises. The wrapper freezes source/scripts before Slurm submission.

CPU resources: one node, four CPU cores, 2 GiB, six minutes. GPU: one CPU core, one GPU, 2 GiB, six minutes. ParaView/profile: one CPU core, 2 GiB, six minutes. No benchmark runs directly on the login/notebook host.

Automated submissions save IDs and resume monitoring on rerun. The notebook waits at most 30 minutes per job by default; reaching the limit never cancels or resubmits a job. Source/settings changes create another run folder.

Local kernel checks: `python tests/check_cpu.py`. Controller checks: `python tests/check_automation.py`. See `tests/automation-validation.md` for what was tested locally and which site/GPU/ParaView checks remain.
