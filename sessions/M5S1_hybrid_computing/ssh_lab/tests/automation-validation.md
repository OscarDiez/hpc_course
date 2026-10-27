# Automated workflow validation

Validated locally on 27 October 2026. This is not SciTech acceptance evidence.

- The complete notebook executed from a clean namespace, without NumPy or Matplotlib required by the default route.
- Actual serial, OpenMP, MPI and hybrid C programs ran with GCC and MPICH in a local scheduler harness. Full-field checks and the deliberate library mismatch passed.
- Actual gprof output was collected.
- Temperature, centre-line and timing SVGs were generated and parsed as XML.
- Saved job IDs prevented duplicate submission. Queue wait/resume, failed accounting, and an ambiguous submission response were tested.
- GPU execution was deliberately failed by the harness to verify honest failure reporting; CUDA/OpenACC were not run.
- ParaView was absent locally and correctly reported SKIPPED. The ASCII VTK exporter was checked for dimensions, point count, units label and exact scalar values. pvpython rendering still requires a site test.
- Notebook schema, Python syntax, Bash syntax, copy/paste source blocks and embedded source parity were checked.

Reproduce the controller test with `python tests/check_automation.py` from this lab directory. It needs IPython, GCC, mpicc and mpirun. It simulates scheduler commands and uses temporary folders. It never claims a simulated Slurm interface is a real cluster allocation.
