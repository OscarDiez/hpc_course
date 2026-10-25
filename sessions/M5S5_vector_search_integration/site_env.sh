# SciTech configuration used in the successful instructor preflight, job 24943.
# Apply inside batch jobs; do not edit shell startup files or the running kernel.
export PATH="/usr/bin:/bin:$PATH"
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export OMPI_CC=/usr/bin/gcc

# Shared Python 3 image: no download, copy, installation or GPU is needed.
export M5S5_IMAGE=/data/software/containers/pytorch-2.8.0-cuda12.8.sif
# Optional C-in-container work requires a separately verified C/OpenMP image.
