# Local validation of the shorter M5S1 notebook

- 53 checks passed against an independent Python heat solver: six grid/step cases, naive and optimised serial, OpenMP, real MPI/hybrid with one to three ranks where the equal-row layout permits them, recorded worker counts, and rejection of an invalid layout.
- GCC 13.3, MPICH 5.0.2; local container transport used `MPIR_CVAR_CH4_NETMOD=ofi FI_PROVIDER=tcp MPIR_CVAR_CH4_SHM=none MPIR_CVAR_CH4_CMA_ENABLE=0` because container restrictions prevent CMA access. These are local test settings, not SciTech instructions.
- Complete notebook code executed with a local Slurm shim and real CPU programs. All 18 field outputs passed, first-step checks passed, real gprof output was collected. The harness deliberately marked the GPU job failed and ParaView unavailable, and the notebook reported them accordingly.
- Notebook schema, Python syntax, Bash block syntax and embedded-file parity checked. OpenACC source compiled with GCC for syntax checking; this does not validate NVIDIA execution.
- No SciTech submission, GPU execution or ParaView rendering was performed for this rewrite. No cluster performance numbers are supplied.

Run the numerical check from an environment with `gcc`, `mpicc`, `mpiexec` and Python:

```bash
python3 tests/check_simple.py
```
