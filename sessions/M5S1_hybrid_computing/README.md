# M5.S1 Hybrid computing basics

Open [M5S1_hybrid_computing.ipynb](M5S1_hybrid_computing.ipynb).

Each C implementation has its own file: serial, OpenMP, MPI and hybrid MPI/OpenMP. The OpenACC solver is an extension. Each source cell is followed immediately by its compiler command, then its execution and checkpoint. There are no conditional-compilation switches.

Each displayed SSH command block has a matching executable notebook cell. Run All follows the same steps, streams output and prints waiting messages when an allocation queues. The notebook needs a connected SciTech kernel. Python sends commands to the shell; all solver and validation work uses C and shell commands.

Checkpoints use the known first-step values, complete-field comparisons with cmp, ownership lines and Slurm output. The text result files have a small VTK header. They are used for recording/comparing temperatures; there is no visualisation stage or rendering dependency.

The CPU batch reuses the previously compiled executables, repeats each three times and creates a separate instrumented profile. The GPU request uses srun --immediate=5: it skips if resources cannot be allocated within five seconds, without leaving a GPU job queued. Execution errors remain visible and separate from a skip.

The runnable files are in [simple_lab](simple_lab). The [extended notebook](M5S1_hybrid_computing_extended.ipynb) preserves the longer earlier material in ssh_lab.

Local tests passed 41 numerical/layout checks with real serial/OpenMP/MPI/hybrid executables. Clean notebook execution was checked with a local scheduler shim and real CPU/profiling work. GPU skip/error handling was tested with mock allocations; NVIDIA execution and immediate allocation on SciTech remain unverified.

Raw notebook URL:

https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m5s1-hybrid-patterns-2026/sessions/M5S1_hybrid_computing/M5S1_hybrid_computing.ipynb

The final results table reports complete CPU field comparisons, the known first-step row, measured loop times, the repeated batch check and a fresh GPU PASS/SKIPPED/FAILED status. It uses the same shell commands in SSH and Jupyter.

The Jupyter shell runner removes inherited Slurm allocation and step variables from its child environment while preserving client configuration and account/QOS preferences. This prevents a four-core hybrid request from being treated as a step inside a smaller notebook job. The notebook session itself is unchanged; srun requests new resources. Batch scripts retain their own fresh Slurm environment for their internal steps.

The C examples use one statement per line, descriptive grid/index/row names, and comments around the numerical update, fixed edges, buffer roles, halo exchanges, threading barriers and final gathering. Output helpers appear after the main calculation.
