# M3S4 — Libraries, I/O and reproducible workflows

Build: **M3S4-2026-10-06-v9**. The notebook embeds all six accompanying source files from `session_demos/14_libraries_io`; downloading the notebook requires no extra source downloads.

## Class route

Run setup → edit SETTINGS → materialize → submit → collect. SciTech uses a fresh Slurm allocation: one node, configurable MPI ranks (default 4), `blas_threads` CPUs per task (default 1), 4 GB and a 10-minute limit. Existing Slurm/MPI queue handling is retained. EESSI/foss/2023b supplies the compiler/MPI/OpenBLAS stack; compatible FFTW/HDF5 modules are loaded when available. The kernel Python executable is retained. Loading HDF5 does not install h5py into that interpreter.

Use a user-owned **shared** WORKSPACE visible to compute nodes. For your own laptop only, set `RUN_LOCAL=True`. Without an execution environment, the notebook prints NOT RUN. If queued, rerun the collection cell. Unchanged submission settings reuse a persisted run; NEW_RUN requests a fresh repetition. Every job freezes source and settings copies.

## Measured exercises

- **§1 FFTW:** compiled C direct DFT versus real serial FFTW, same input/normalization, NumPy correctness check. Planning separate from warm repeated execution; editable sizes include non-powers of two.
- **§2A BLAS:** seeded dense matrix multiplication with two compiled C baselines (`ijk` and contiguous `ikj`) versus actual CBLAS DGEMM. All full results checked with tolerance. Optimized compilation, warmup, repeated rotated order, input generation/validation outside timing. Library path/configuration and both baseline speedups are reported. One-thread comparison first; changing `blas_threads` to 2 requests sufficient Slurm CPUs and adds a controlled OpenBLAS thread experiment. System BLAS may be slower than an optimized manual loop. Configured thread count is not evidence of active worker utilization.
- **§2B LAPACK:** fixed-end heated rod `Ax=b`, compiled C partial-pivot Gaussian elimination versus real DGESV. Residual and analytic temperature profile checks; input copies outside timing. A zero-leading-pivot example explains row swapping, and a singular system demonstrates nonzero INFO. Editable resolution, heating and end temperatures. Explicit column-major/LP64 interface. A tridiagonal solver would be better for a production rod solver; dense DGESV teaches the interface.
- **§2D checkpoint:** uninterrupted heat reference, controlled process failure and independent restart process. State includes grid, step, parameters and integrity hash; final arrays match exactly. Atomic rename is not a guarantee of power-loss durability.
- **§4 I/O:** identical ordered payload in many files versus one file, alternating order, raw timings and full SHA-256 checks. Optional fsync. Single writer and warm cache, not a disk-bandwidth or filesystem-scaling benchmark.
- **§6 HDF5:** both Python and C backends support editable time/y/x dimensions, one-plane chunks, Kelvin units, time axis, gzip compression and selected-slice readback. Backend detection tries h5py, h5cc/h5pcc (including the loaded module's bin directory), then EBROOTHDF5 headers/shared library with the compatible compiler. Found-but-broken compilation fails visibly. The C program is a serial writer even if compiled against parallel HDF5. NetCDF/CF conventions are explained as context, not claimed as executed.
- **§7 MPI-IO:** real ranks with disjoint byte ownership, collective write, sync/close, reopen/readback. MPI return codes/counts, every record, global success and complete file are checked. A small one-node example does not demonstrate storage scaling.
- **§8–10 evidence:** separate input/noise seeds, manifest with source/input/output hashes, commands, software, libraries, modules, allocation, filesystem and raw timings. Fresh-process replay verifies input/source identity and numerical tolerance; a damaged copy is deliberately rejected. One-change comparisons now include BLAS and LAPACK evidence.

The compute/I/O fraction is a mathematical model. Containers, workflow tools, sparse libraries and NetCDF remain discussion material. PASS, FAIL and SKIPPED are distinct. Core success requires I/O/checkpoint/replay and no experiment marked FAIL; missing optional dependencies stay SKIPPED. Core success never means every demonstration ran.

## Validation of this revision

- All 19 notebook code cells executed locally, including source materialization, frozen-run execution, compiled C/CBLAS/LAPACK/FFTW, checkpoint restart, payload validation, manifest replay and tamper rejection. HDF5 h5py full/slice readback passed with default and edited/gzip settings.
- System LP64 BLAS/LAPACK passed default numerical checks. Real optimized OpenBLAS also passed one/two configured thread comparisons and DGESV checks; its test wheel's prefixed symbols were aliased only in the local test harness. Production uses the standard OpenBLAS symbols supplied by the module stack.
- Edited rod sizes 2, 37 and 256, zero/positive/negative heating and changed boundaries agreed with the analytic solution. Zero-pivot and singular cases returned expected results/status.
- C HDF5 compiled with `-Wall -Wextra -Werror` against real HDF5 1.10.10. Default and edited/gzip cases passed full field/time-axis and hyperslab checks; both loaded-module direct compilation and h5pcc-wrapper discovery were exercised. h5py independently inspected the resulting C-written dataset. The SciTech module stack still requires a fresh cluster run.
- The preserved MPI-IO implementation previously passed an actual four-rank SciTech run (user execution, build v6). This revision has not been run on SciTech; local environments that cannot launch MPI report SKIPPED. Do not treat previous cluster evidence as an execution of v7.

## Student changes

Start with the default run. Predict an outcome, change one setting, rerun setup onwards, save the run ID and compare. Useful first changes: double one BLAS size, set BLAS threads to 2, set rod heating to zero, enable gzip, change checkpoint interval, or increase file count while discussing whether total bytes also changed. Retain the manifest with submitted evidence.


## v8 runtime dependency fix

The C HDF5 compiler and executable now receive an explicit child-process library environment from existing LD_LIBRARY_PATH/LIBRARY_PATH and the currently loaded EasyBuild module roots (including Szip/libaec), supplemented by dependency -L paths reported by the selected HDF5 wrapper. This handles indirect compression dependencies that an executable RUNPATH alone may not resolve. No system installation or package download is required on SciTech. The HDF5 report records runtime directories and loader dependency resolution; unresolved dependencies remain FAIL with actionable diagnostics.

Validation: reproduced the exact libsz.so.2 runtime failure using a real HDF5 library outside system paths, then passed full/slice readback with only loaded-module roots supplying the missing Szip/libaec paths. Default and edited/gzip C cases passed. Fresh SciTech v8 execution remains to be confirmed.

## v9 FFTW loading and stale-run checks

FFTW discovery now tries absolute shared-library paths from the selected EBROOTFFTW module before generic SONAME lookup; all discovered load failures remain visible. Collection prints both notebook and report builds and rejects stale report builds or source/settings fingerprints. Recollecting an old run does not execute a corrected source version.

Validation: deliberately inaccessible FFTW SONAME reproduced a loader failure, while the selected module absolute path passed DFT and NumPy spectrum checks for sizes 128 and 250. All 19 notebook cells also ran with actual C HDF5 including repaired compression paths; old report builds and stale RUN_STATE fingerprints were rejected. Fresh SciTech v9 execution remains to be confirmed.

## Student-facing activities

Each activity states the scientific situation, what actually executes, what result to inspect, one setting to change, and the concept to explain. Setup runs the measured worker; numbered activity cells display saved evidence. Replay/tamper cells explicitly start new verification processes. Discussion/model sections are labeled as such.

Checkpoint presentation shows the three-process reference/failure/restart sequence, a saved-field plot, saved/lost/restarted work, and a clearly labeled prediction table. The failing step is retried in addition to the reported lost previously completed steps. BLAS/FFTW/LAPACK/I/O use labeled timing and correctness tables. HDF5 plots identify whether data came from direct file reading or the C-verified expected slice. MPI shows actual rank ownership and launch evidence. Detailed source/raw reports remain expandable. The one-change comparison displays relevant measured quantities from two reports and checks matching scientific source hashes.

Validation: all 19 code cells executed locally, measured checkpoint interval 10→5 comparison confirmed saved step 50→55 and lost completed work 6→1, checkpoint plot inspected, and ownership display checked against the user's actual four-rank SciTech result. Scientific worker/source fingerprints remain v9 and unchanged. User's v9 SciTech export reported all eight experiments PASS and no skips; this presentation revision can reuse that valid measured run.
