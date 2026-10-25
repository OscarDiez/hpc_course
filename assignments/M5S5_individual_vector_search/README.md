# M5S5 · Individual cluster assignment: batched top-k search

Read [the student brief](STUDENT_BRIEF.md). This is separate from the guided notebook, module quiz and group milestone.

Starter files:
- `topk_baseline.c`: correct serial full-sort baseline, not an optimized solution.
- `check_topk.py`: strict all-query/all-K reference comparison.
- `baseline.sbatch`: baseline-only job; add your own parallel builds/runs/checks.
- `site_env.sh`: tested system compiler/MPI configuration and shared Python SIF, sourced inside batch jobs.
- `container_replay.py`: unchanged portable validator/analysis payload for native Python and Apptainer.

Required work includes real C/OpenMP searches on Slurm, correctness fixtures, three-repeat scaling, a measured optimization, Q sensitivity, hybrid execution or the evidenced CPU alternative, and a **required Apptainer compute-output validation workflow**. The Python image does not need gcc; the C computation runs natively. Students practise binds, environment isolation/forwarding, numerical comparisons, hashes, deliberately failed validation, and a reproducible shell workflow. C/MPI inside the image is not assumed. No GPU or installation is required.

Main workload: N=65536, D=128, Q=32, K=10. Include one additional student-designed diagnostic test. Submit the assignment evidence questions and one self-contained PDF with code/scripts/results and environment/container evidence. Course weights and deadlines remain instructor settings.
