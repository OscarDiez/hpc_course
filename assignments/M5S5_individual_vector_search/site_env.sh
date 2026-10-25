# SciTech configuration validated in instructor preflight job 24943.
# Source inside batch jobs, before building/running the assignment.
export PATH="/usr/bin:/bin:$PATH"
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export OMPI_CC=/usr/bin/gcc

# Centrally installed Python 3 image. No download, copy or GPU required.
export M5S5_IMAGE=/data/software/containers/pytorch-2.8.0-cuda12.8.sif
