# M5.S1 Hybrid computing

Start with [M5S1_hybrid_computing.ipynb](M5S1_hybrid_computing.ipynb). This is the shorter student practical: 29 cells, two C programs, direct compile/run commands and six checkpoints.

The sequence is heat diffusion, a serial baseline, removal of array copying, compiler optimisation, OpenMP, MPI, hybrid MPI/OpenMP, measurements, profiling and ParaView. OpenACC is a GPU extension using the same heat source. A larger experiment asks students to predict and explain changes.

Students work in SSH. A separate teaching demonstration at the end can reproduce the practical through Run All on SciTech JupyterHub. It submits one CPU job and one OpenACC job, retains their IDs, checks complete fields and displays a ParaView screenshot only after a successful render. The default CPU request is one node, two tasks, two cores per task, 2 GiB and six minutes. The GPU request is one task and one GPU. Jobs may queue.

The five files shown in the notebook are also in [simple_lab](simple_lab): `heat.c`, `heat_mpi.c`, `run.slurm`, `gpu.slurm` and `view.py`. There are no separate build, submission, measurement or plotting frameworks in this route. Module names match the observed SciTech environment; inspect `module avail` if availability changes.

The [extended notebook](M5S1_hybrid_computing_extended.ipynb) preserves the previous full material, including CUDA, the library compatibility exercise and additional slide demonstrations. Its supporting files remain in `ssh_lab`. It is optional reference material.

Local validation of the shorter version passed 53 numerical, ownership and invalid-layout checks with GCC/OpenMP and real MPICH processes. The complete notebook also passed a local scheduler harness using real CPU/MPI/OpenMP/profiling execution. The scheduler was mocked; GPU execution and ParaView rendering were not verified. These tests are not measured SciTech results.

Raw notebook URL for Jupyter:

https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s1-hybrid-patterns-2026/sessions/M5S1_hybrid_computing/M5S1_hybrid_computing.ipynb
