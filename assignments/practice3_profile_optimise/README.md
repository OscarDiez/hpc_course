# Practice 3 — Profile and Optimise

This directory contains the starter pack for Module 3 Practice 3.

## Files

- `src/p3_pipeline_student.c` — baseline program + one optimisation TODO
- `jobs/p3_baseline.sbatch` — correctness, repeated baseline, gprof, GNU time, optional perf
- `jobs/p3_optimised.sbatch` — repeated changed run + reprofile
- `results_template.csv` — table for measured evidence
- `make_evidence.sh` — generates `p3_evidence.txt`

## Core rule

Do not edit the optimisation TODO until you have collected the baseline, inspected the profile and written your bottleneck hypothesis.

Change exactly one factor: fuse the three full-array passes inside `pipeline_optimised()`.

Keep the formula, input, steps, compiler flags and Slurm resources unchanged.

Blackboard contains the definitive assessed instructions and submission requirements.
