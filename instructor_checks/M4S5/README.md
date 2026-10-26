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

## Confirmed assignment preflight results: jobs 24984 and 24985

Instructor-provided console output confirms both jobs COMPLETED, exit code 0:0, elapsed 43 seconds each. Configurations: N=4096, epochs=6, seed=3068; global batches 128 and 256 respectively. Both passed the supplied baseline/C/real DDP/candidate/recovery/inference check and the intentional unequal-shard error detection. Overall PREFLIGHT ONLY PASS; student correction/full-training comparison intentionally NOT RUN. This supersedes the earlier pending status for these two jobs.

Fresh checks after the student/instructor file separation were submitted as jobs 24986 and 24987 in `check.30kJA0/b128` and `check.30kJA0/b256`. Their completed summaries have not yet been supplied. Do not infer success from submission.
