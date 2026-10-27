# M5.S1 Hybrid computing basics

Open [M5S1_hybrid_computing.ipynb](M5S1_hybrid_computing.ipynb).

Each implementation has its own complete C file: serial, OpenMP, MPI and hybrid MPI/OpenMP. There are no conditional-compilation switches. The OpenACC heat solver is an optional extension. Read the stencil first and follow the compile/run commands beside each program.

Students can create the files with an SSH editor or execute the notebook's `%%writefile` cells. The solvers always run as C executables on allocated resources. Python does not simulate the plate or validate fields. Checkpoints use `sed`, `cmp`, Slurm commands and, for GPU rounding differences, an `awk` comparison. Every solver writes VTK directly.

The final Jupyter section sends the same shell commands through a small subprocess wrapper. It submits the CPU and OpenACC jobs with `sbatch --wait`, displays their logs and the available ParaView screenshot. Each invocation submits new jobs. The separate optional ParaView script uses ParaView's own Python API only to open the C output and save an image.

The files shown in the notebook are in [simple_lab](simple_lab). One CPU batch file compiles and compares the four CPU versions, repeats each three times and saves a profile. One GPU batch file runs OpenACC and checks it against a CPU reference. There is no build, submission or measurement framework.

The [extended notebook](M5S1_hybrid_computing_extended.ipynb) preserves the longer earlier CUDA, library and slide demonstrations, with its files in `ssh_lab`.

Local tests passed 41 numerical/layout checks with real serial/OpenMP/MPI/hybrid executables. The whole notebook also executed with a local scheduler shim and real CPU/profiling runs. The shim deliberately fails the GPU request and makes ParaView unavailable; these outcomes are reported rather than shown as successes. SciTech execution, NVIDIA execution and ParaView rendering remain unverified for this rewrite.

Raw notebook URL:

https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s1-hybrid-patterns-2026/sessions/M5S1_hybrid_computing/M5S1_hybrid_computing.ipynb
