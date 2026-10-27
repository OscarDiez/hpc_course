# M5S1 SSH heat laboratory

The first part of `../M5S1_hybrid_computing.ipynb` is the complete student guide. Existing notebook cells remain in the middle. The last part replays this lab with a self-contained source bundle.

CPU: `N=512 STEPS=500 REPEATS=3 bash scripts/submit.sh cpu all`

GPU: `N=512 STEPS=500 REPEATS=3 bash scripts/submit.sh gpu all`

Run from this directory on the SciTech login node. The wrapper freezes source/scripts in a new directory and submits Slurm; no benchmark executes on login. Select `serial`, `openmp`, `mpi`, `hybrid` or `libraries` instead of `all` for one stage.

Resources: one CPU node, four CPU cores, 2 GiB, six minutes; or one GPU, one CPU core, 2 GiB, six minutes. Verify site module names in scripts/site.sh. Default launcher srun; set MPI_LAUNCHER=mpirun only if supported by the site. Communicator-size mismatch fails explicitly.

Local checks: `python3 tests/check_cpu.py` requires GCC/OpenMP/Python and optionally mpicc/mpirun. They compare full fields with an independent Python stencil and check unequal decomposition and library semantics. Local development checks are separate from SciTech/GPU acceptance.
