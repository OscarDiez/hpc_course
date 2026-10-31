# Validation — M6S1 neuromorphic notebook

2026-10-31. Local CPU; GCC 13.3.0.

- PASS: all eight plain-Python code cells executed in order using a Python harness, capturing real saved output. Python syntax parsed. This was not a Jupyter frontend test.
- PASS: both embedded C programs compiled with C11, `-Wall -Wextra -Werror`.
- PASS: independent Python neuron recurrence agrees with C for threshold/retention pairs (1,0.8), (1.5,0.8), (1,0), (1,0.5), (1,0.9): spike counts 4,2,0,2,4.
- PASS: initial plus two change messages reconstruct all eight baseline samples.
- PASS: alternating-input extension reconstructs all eight samples using eight messages.
- NOT RUN: SciTech scheduling, modules, compute-node runs, JupyterLab URL import, and optional external Python/Colab exercises.
- SKIPPED: neuromorphic chip execution; CPU teaching simulation only. No runtime/energy comparison measured.

The notebook includes explicit missing-GCC and failed-compilation/execution handling; these error branches were inspected, not fault-injected in this revision. Queue time is separate from simulation work. The saved notebook uses default inputs; student changes can intentionally invalidate baseline assertions, as explained alongside the task.
