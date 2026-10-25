# M5S5 · Individual cluster assignment: batched top-k search

Read [the student brief](STUDENT_BRIEF.md) first. This is an individual assessed task, separate from the guided notebook, module quiz and group milestone.

Provided files:
- `topk_baseline.c`: correct serial full-sort baseline; no parallel/optimized solution is supplied.
- `check_topk.py`: all-query, all-neighbour ordered correctness comparison.
- `baseline.sbatch`: small CPU allocation running only baseline repetitions; adapt for your own implementation.
- `site_env.sh`: configure the actual site modules here.

Required main input: N=65536, D=128, Q=32, K=10. The guided notebook uses top-1 and Q=16 instead. Keep the two activities' results distinct.

Blackboard assessment: eight evidence questions (40 internal points) and one self-contained evidence PDF (60). Questions assess interpretation; PDF contains code, scripts, measurements, checks and logs needed for review. Deadline and course-grade weighting are configured separately by the instructor. No module quiz is included here.

Use the [cluster guide](../../cluster/README.md). All significant work goes through Slurm. No GPU, package installation or large dataset is required. The documented MPI/CPU-alternative and container/module-alternative routes accommodate actual availability without inventing results.

Validation: starter compiled with GCC 13, C11, -O3, -Wall -Wextra -Werror locally. Independent Python full-top-k checks passed normal, exact ties, N=1 and K=N fixtures. Checker rejected wrong final neighbour, missing query, duplicate ID and wrong score. Three default-size serial runs completed locally; these are not SciTech measurements. Batch syntax passed. Student parallel/optimized implementations do not exist in this starter and have not been validated. SciTech launch, modules and container compatibility remain NOT RUN.
