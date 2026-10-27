# Local checks for M5S1 basics

41 checks passed against an independent Python reference in the test code: six grid/step cases, serial, OpenMP, MPI/hybrid with one to three ranks for compatible layouts, correct VTK dimensions/values, ownership labels and rejection of an invalid layout. The student notebook itself contains no Python heat simulation or numerical validator.

All notebook code cells executed with IPython, including the writefile magics. A local scheduler shim ran the actual CPU batch commands, GCC/OpenMP/MPICH programs, field comparisons and gprof. The shim deliberately failed the GPU job and made ParaView unavailable. The notebook displayed those outcomes and did not fabricate an image.

Local environment: GCC 13.3, MPICH 5.0.2. Container-only MPI settings were `MPIR_CVAR_CH4_NETMOD=ofi FI_PROVIDER=tcp MPIR_CVAR_CH4_SHM=none MPIR_CVAR_CH4_CMA_ENABLE=0`, because CMA access is restricted. These settings are not student or SciTech instructions.

OpenACC source compiled with GCC for syntax checking. This is not NVIDIA execution. No SciTech job or actual ParaView render was run for this rewrite. Local timing values are not presented as cluster measurements.

With gcc, mpicc, mpiexec and Python available, run:

```bash
python3 tests/check_basics.py
```

The current step-by-step notebook was executed from a clean IPython namespace with the local scheduler shim. Direct serial/OpenMP/MPI/hybrid srun commands and the later repeated batch executed real programs. Each displayed SSH block matches its executable notebook cell. ParaView and GPU requests deliberately failed in the harness; errors were reported and the final review still ran. The 41 numerical checks still cover the unchanged C sources.
