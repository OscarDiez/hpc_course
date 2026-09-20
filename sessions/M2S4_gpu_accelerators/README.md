# M2.S4 - Introduction to GPU and Accelerator Computing

Interactive notebook for **M2.S4 - Introduction to GPU and Accelerator Computing**.

## Open in IE/SciTech JupyterHub

1. Open: https://jupyter.iesci.tech:8085/
2. Choose **File → Open from URL...**
3. Paste:

```text
https://raw.githubusercontent.com/OscarDiez/hpc_course/main/sessions/M2S4_gpu_accelerators/M2S4_gpu_accelerators.ipynb
```

4. Use the **Python 3** kernel.
5. Save a personal copy if desired.

## Execution model

- Small conceptual activities run directly in Jupyter.
- The real accelerator experiment submits **one Slurm GPU job**:
  - `--partition=gpu`
  - `--gpus=1`
- The GPU job always runs `nvidia-smi`.
- CUDA tries native `nvcc`, the advertised module routes (`hpc-course-cuda/12.8.1` and `nvhpc/25.7`), then the shared Apptainer image at `/data/software/containers/hpc-course-cuda.sif` as a fallback. Current validation confirms `nvhpc/25.7` works.
- OpenACC tries native `nvc`, then `nvhpc/25.7`.
- The real experiment compares the same vector-add workload on CPU and GPU, including GPU kernel-only and end-to-end timings.
- If no compiler route is exposed, the job prints diagnostics suitable for cluster-admin follow-up.
- An optional CuPy check is included if the library is installed.

## Classroom activities

1. environment / scheduler check
2. classify GPU-friendly workloads
3. CUDA grid → blocks → threads
4. model data-movement overhead
5. inspect a CUDA vector-add program
6. inspect an OpenACC offload version
7. **real SciTech GPU job**
8. CPU vs GPU end-to-end timing
9. library vs OpenACC vs CUDA decision
10. final bottleneck challenge

Legacy GPU material remains under `chapters_examples/` and `session_demos/09_gpu/`.
