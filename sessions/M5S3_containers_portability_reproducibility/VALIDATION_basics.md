# Checks for the modules and Apptainer practical

The notebook schema, code-cell Python syntax, generated Bash syntax, source/configuration parity and all nine matching SSH/notebook command blocks passed.

All notebook code cells executed through IPython, including writefile cells, in three fresh local scenarios:

| Scenario | Actual payload work | Orchestration tested |
|---|---|---|
| Compatible CLI test double | GCC compiled the real C program; its actual sum was 5000050000. Host CPU PyTorch 2.8.0+cpu trained A, B, C and the mounted-path repetition. A/B and A/bind records matched, C changed and numerical assertions passed. A real host file was written/read. | Mock Lmod, Slurm and Apptainer CLI; command arguments, cleanenv/bind construction, status handling and final table |
| Missing image | Real C calculation passed. No training process was launched. | Container SKIPPED; repeat/change/bind SKIPPED; final summary completed |
| Incompatible runtime | Real C calculation passed. The deliberately incompatible CLI returned failure. | Container FAILED; dependent training/change/bind SKIPPED; final summary completed |

The runner was tested with inherited Jupyter job ID 25204 and a one-core setting. Its child requests lost those allocation limits, while the parent environment remained unchanged. Test allocation arguments requested CPUs only; no GPU or package/image download command is present in the student workflow.

These are local author checks, not SciTech execution. Lmod save/restore, Slurm allocation, Apptainer isolation, the shared SIF, image-specific interpreter and actual filesystem bind semantics are not validated by a test double. The local PyTorch build was CPU-only; it is not claimed to be the shared cluster image. No cluster timing or container result is prefilled in the notebook.

Run the local orchestration test in an author environment with nbformat, IPython, GCC and CPU PyTorch:

```bash
python3 basic_lab/tests/check_basics.py
```

For acceptance on SciTech, use a fresh notebook copy, run in order, inspect actual module paths, CPU allocations, PyTorch/image information, all checkpoint statuses and the final table. Preserve a genuine FAILED or SKIPPED status. The user must see actual native and container outputs before the container route is treated as demonstrated.
