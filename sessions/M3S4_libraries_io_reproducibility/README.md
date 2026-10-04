# M3S4 — Libraries, I/O and reproducible workflows

Build: **M3S4-2026-10-04-v5**. The notebook embeds all five accompanying files from `session_demos/14_libraries_io`; a downloaded notebook needs no source downloads.

## Class route

Run setup → edit SETTINGS → materialize → submit → collect. By default, SciTech uses one Slurm job with one node, configurable MPI ranks (default 4), 4 GB and a 10-minute limit. EESSI/foss/2023b selects the compiler/MPI stack. Compatible visible FFTW/HDF5 modules are loaded when available. The notebook Python interpreter is retained, so an HDF5 module does not automatically install h5py into that kernel.

A user-owned **shared** WORKSPACE must be visible to compute nodes and the notebook. A local laptop requires explicit `RUN_LOCAL=True`; do not enable that on a shared login node. Without an execution environment the notebook prints NOT RUN, not invented timings. If a job remains queued, rerun only the collection cell. Unchanged submission settings reuse a persisted job/run; NEW_RUN requests a new repetition. Every job freezes its own source and settings copies.

## Actual exercises and their scope

- Sensor spectrum: compiled C direct DFT and actual serial FFTW, same input and forward normalization, independent NumPy correctness check. Plan creation is separate from repeated warm execution. FFT batch timings include amortized Python binding overhead. Editable transform sizes include non-powers of two.
- Heat checkpoint: uninterrupted reference, controlled process termination and a separate restart process. Saved state includes current grid, step, parameters and integrity hash. Final arrays must match exactly. The exercise measures checkpoint write count/time and lost completed steps; it does not promise power-loss durability or distributed checkpointing.
- I/O: identical ordered records in many files versus one file on the recorded run filesystem, alternating order, raw samples and full payload SHA-256 verification. `fsync` is optional. This is a single-writer/cache-sensitive exercise, not a Lustre scalability or disk-bandwidth benchmark.
- Structured weather field: editable serial h5py/HDF5 dimensions, units, chunks, compression and selected time slice; reopen and verify all data. A C HDF5 fallback uses fixed 4×8×8 dimensions and a checked hyperslab. Editable shape/compression options require h5py. Serial HDF5 is not parallel HDF5.
- MPI-IO: real MPI ranks, explicit disjoint byte ownership, collective write, sync/close, reopen and collective readback. Verify MPI return codes, counts, every record, global success and the complete file independently. A tiny one-node correctness example does not prove filesystem scaling.
- Evidence: independent input and noise seeds, manifest with exact source/input/output hashes, commands, software, modules, allocation, filesystem and raw measurements. A fresh-process replay checks input/source identity and numerical tolerance; altered input is deliberately rejected.
- The compute/I/O fraction cell is explicitly a mathematical model. BLAS/LAPACK, sparse libraries, NetCDF, containers and workflow tools are discussed as context; the notebook does not claim to have executed every named library or built a container.

PASS, FAIL and SKIPPED are distinct. Core success requires I/O, checkpoint and replay to pass and no executed experiment to fail. Unavailable optional FFTW, HDF5 or MPI dependencies remain listed as SKIPPED; core success does not imply that all demonstrations ran.

## Validation performed before publication

- All 17 notebook code cells executed end-to-end locally with actual FFTW and h5py; MPI was explicitly unavailable in that run. All 17 also executed in no-Slurm/no-local-opt-in mode without producing fabricated benchmark results.
- FFTW spectra agreed with the C DFT and NumPy for default and non-power-of-two sizes; default strongest bins were 3 and 7.
- Equal-payload I/O checks passed with both close-only and fsync modes and small nondefault records.
- Separate heat processes passed restart checks for intervals 5/10/25, failure on a checkpoint boundary, and failure before the first checkpoint. Corrupt checkpoint state was rejected.
- Editable HDF5 shape, selected slice and gzip compression passed full readback checks. Replay passed in a fresh process and deliberately altered input failed its checksum check.
- MPI C source compiled with `-Wall -Wextra -Werror` and passed real single-rank collective write/read verification under MPICH. A four-rank local launch was blocked by this execution environment's socket/shared-memory restrictions; it was **not** reported as verified. Four-rank SciTech execution and the site's C HDF5 wrapper fallback remain to be verified on that cluster.
- Submission/collection guard checks exercised persisted job reuse, queued results, missing reports and failed experiments using controlled mocks. These guard checks are not cluster execution evidence. Batch shell syntax was checked with `bash -n`.

The notebook installs no packages. Test-only dependencies used by the author were isolated outside the notebook/repository.
