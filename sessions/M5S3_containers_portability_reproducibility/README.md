# M5.S3 Modules and Apptainer

Open [M5S3_containers_portability_reproducibility.ipynb](M5S3_containers_portability_reproducibility.ipynb).

Each step says what to run and why, shows the direct SSH commands and provides a notebook cell executing those same commands. Run All follows the sequence. There are no status files, conditional shell branches or image-discovery framework in the student route. Actual command failures remain visible.

The route is module inspection, immediate C compilation and an allocated run, module save/purge/restore, HDF5 wrapper inspection, inspection of the existing Apptainer image, PyTorch runs A/B/C, a short bind example, and a two-server Slurm launch. The final table checks the saved results.

The bind example has one small source file, one container execution and a host cat command. It reads the mounted folder and writes a note that remains on the host. The two-server command requests two CPU nodes and one container process on each with srun -N2 -n2 --ntasks-per-node=1. Each process prints its server name; no MPI or distributed training is implied. The allocation can queue until two nodes are available.

The four source files in [basic_lab](basic_lab) are all used: module_demo.c, train_model.py, write_note.py and mpi_demo.c. No container.env file or generated worker script is needed. The source/configuration paths use the versions actually observed in the instructor's SciTech export: foss/2023b, HDF5/1.14.3-gompi-2023b and the shared PyTorch SIF at /data/software/containers/pytorch-2.8.0-cuda12.8.sif, with its /opt/conda/bin/python3 interpreter. No package or image installation is performed.

The notebook interpreter does not need PyTorch. Slurm allocates resources, and the container supplies the Python/PyTorch user-space environment. The small Jupyter shell runner removes inherited Slurm allocation limits from its child environment so the exercises can request their own resources.

The [extended notebook](M5S3_containers_portability_reproducibility_extended.ipynb) preserves the older full dependency, cache, library and handover material. Existing examples and their validation remain unchanged. validate_notebook.py points at that extended version.

See [VALIDATION_basics.md](VALIDATION_basics.md) for local checks and the observed SciTech results. The new bind-note and two-node examples still need actual cluster confirmation.

Raw notebook URL:

https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s3-reproducibility-2026/sessions/M5S3_containers_portability_reproducibility/M5S3_containers_portability_reproducibility.ipynb

The last exercise adds actual MPI communication: two ranks contribute 1 and 2 to MPI_Allreduce and must both receive 3. Slurm salloc reserves two CPU nodes; host mpirun maps one rank/container to each node. The host MPI and dependencies are exposed through a read-only CVMFS bind, LD_LIBRARY_PATH and OPAL_PREFIX. MPI launch variables are preserved, so this command does not use cleanenv. Ordinary TCP is selected for the small example. This new container/MPI launch still needs SciTech confirmation. Basic results appear before it, and its own final check requires ranks 0/2 and 1/2, distinct server names and sum 3.
