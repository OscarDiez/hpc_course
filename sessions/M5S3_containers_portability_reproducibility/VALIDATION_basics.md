# Validation of the direct modules/Apptainer practical

## Observed SciTech results

The instructor-provided M5S3 HTML export (11) confirms:

- foss/2023b selected GCC 13.2.0 and OpenMPI 4.1.6. Native job 25223 printed the correct sum 5000050000.
- The m5s3_class collection saved, purged and restored; compiler paths matched.
- HDF5/1.14.3-gompi-2023b loaded and h5pcc was found.
- Apptainer 1.5.4 opened the shared 4.0 GiB PyTorch image. Job 25224 imported PyTorch 2.8.0+cu128 using /opt/conda/bin/python3. CUDA was unavailable, which is acceptable for the CPU-only exercise.
- Jobs 25225 and 25226 produced identical A/B JSON records: target slope 3, learned slope 2.996894, intercept 1.999999, mean absolute error 0.03219.
- Job 25227 used target slope 4 and returned learned slope 3.996894 with intercept 1.999999 and error 0.03219.
- The previous bind experiment was queued as job 25228 when the export ended. It is not evidence of a successful bind.

## Current local checks

The student route now uses direct commands, without status-file gates or conditional Bash branches. All notebook code executed through IPython with actual GCC and host CPU PyTorch 2.8.0+cpu payloads, using explicit local test doubles for module, srun and apptainer commands.

The local compatible case verified the known C sum, module-path comparison, model assertions, identical A/B records, changed C and the host note. The two-node test double checked command arguments and final distinct-hostname parsing; it did not run on two machines. Missing-image and incompatible-runtime cases stopped at the actual failed command rather than reporting a fabricated success. No GPU was requested.

Schema, Python/Bash syntax, source parity, readable line continuations and matching SSH/Jupyter commands were checked. Run the author test with nbformat, IPython, GCC, an MPI C compiler and CPU PyTorch available:

```bash
python3 basic_lab/tests/check_basics.py
```

The new bind-note program and two-node command require actual SciTech execution. A test double does not validate Apptainer isolation, filesystem mounts, distributed placement or hardware availability. No new cluster results or timings are prefilled in the notebook.

The final MPI source compiled and ran with real MPICH 5.0.2 in two local processes; both received sum 3. This is not a two-server or container test. The full notebook CLI harness also covers salloc/mpirun command construction and final parsing using labelled test doubles. The host-MPI/CVMFS bind, Open MPI runtime compatibility and network launch on SciTech remain unverified. The main results table is produced before this new MPI exercise.
