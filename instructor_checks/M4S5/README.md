# M4S5 instructor checks

Student starter files and instructions are under `assignments/practice4_distributed_ml/`. They contain student baseline/full runs and required correctness checks, not instructor verification scripts.

## Instructor assignment preflight

Run from SciTech SSH:

```bash
mkdir -p "$HOME/m4s5-assignment-validation"
cd "$HOME/m4s5-assignment-validation"
curl -fL 'https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m4s5-distributed-ml-2026/instructor_checks/M4S5/verify_assignment.sh' -o verify_assignment.sh
bash verify_assignment.sh
```

The launcher prepares two student work directories and submits baseline jobs at global batch 128 and 256 sequentially. Its numerical checks exercise supplied code and intentional bad-result rejection; an unfinished student correction remains NOT RUN. It supplies no student solution. Instructor validation records are in `assignment_validation/`.

Previously submitted jobs **24984** and **24985** used the earlier launcher and already-created directories. Moving repository files does not change those queued/running jobs. Their completion and summaries are not yet provided; do not record them as PASS from submission alone.

Original summary commands:

```bash
sacct -j 24984,24985 --format=JobID,State,ExitCode,Elapsed
cat /home/odiez/m4s5-assignment-validation/check.pOT2ix/b128/ASSIGNMENT_SUMMARY.txt
cat /home/odiez/m4s5-assignment-validation/check.pOT2ix/b256/ASSIGNMENT_SUMMARY.txt
```

The guided notebook is separately validated by SciTech job 24983. The notebook verifier remains `verify_cluster.sh` in this directory. A baseline/negative-test PASS does not validate a student's future correction or establish authorship.
