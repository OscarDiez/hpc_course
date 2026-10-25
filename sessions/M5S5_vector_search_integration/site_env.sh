# Put the exact SciTech module load commands here, after checking module avail/list.
# Native CPU: Python 3 and gcc with OpenMP.
# Hybrid: a matching compiler + MPI stack; inspect mpicc --showme and its paths.
# Do not combine a system OpenMPI wrapper with an unrelated EESSI compiler stack.
# Required container workflow: load Apptainer if needed and name an existing trusted Python 3 SIF.
# export M5S5_IMAGE=/absolute/path/to/approved-python3-image.sif
# No GCC or MPI is needed inside that image for the result-validation workflow.
# Optional C-inside-container extension needs a separate gcc/OpenMP-capable image.
