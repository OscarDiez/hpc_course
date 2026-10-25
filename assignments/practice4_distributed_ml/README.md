# Practice 4 - Individual distributed ML assignment

Student-facing brief: `BLACKBOARD_ASSIGNMENT.md`. It has Blackboard evidence questions (40%) and one consolidated PDF (60%). The question bank and slides are separate deliverables.

## Student starter

`setup_assignment.sh STUDENT_ID NEW_DIRECTORY` downloads the pinned, cluster-validated guided notebook at commit `616f5fbe63e2e0d63fbecbcae0232defd44d72ce`, extracts its actual executable source literals without submitting notebook jobs, and applies checked, single-occurrence transformations to add the unequal-shard exercise. No complete correction is supplied: `student.py` intentionally raises NotImplementedError.

`prepare_assignment.py` writes personalized `config.json`, provenance, native C, ML source and two CPU batch scripts. `run_preflight.sbatch` checks supplied capabilities; `run_full.sbatch` also calls the student's function and compares corrected full training with the one-worker reference. No package installation, GPU or image download is used. The shared image is `/data/software/containers/pytorch-2.8.0-cuda12.8.sif`.

The exercise retains identical global-batch membership for every update. For two ranks, balanced shards have half the batch each; unequal shards have three quarters and one quarter. Both ranks still participate in every optimizer update. Naive unequal mean-loss averaging is an intentionally wrong computation; its expected FAIL must be detected with synchronized replicas.

The preflight executes N=4096, six epochs, global batch 128, three repeats each of one-worker and balanced two-worker DDP, the three-candidate workflow, actual interruption/restart, inference and the intentional unequal-shard negative test. Its PASS means only supplied prerequisites passed; student implementation remains NOT RUN. Full mode adds a corrected one-step comparison and three corrected complete runs. Full mode must be run separately at global batch 128 and 256. It does not assess report interpretation or authorship.

Run one job per work directory at a time. Generated outputs from a previous run are preserved in `previous_outputs/`; existing Slurm logs also remain distinguishable by job ID. Use separate directories for the two batch configurations. Do not confuse an old summary with a newly pending job; match the report job ID to `sacct`.

## Instructor test

From SciTech SSH:

```bash
mkdir -p "$HOME/m4s5-assignment-validation"
cd "$HOME/m4s5-assignment-validation"
curl -fL 'https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m4s5-distributed-ml-2026/assignments/practice4_distributed_ml/verify_assignment.sh' -o verify_assignment.sh
bash verify_assignment.sh
```

This makes fresh directories, prepares the actual student starter with representative identifier 1042 (seed 3068), and submits the exact preflight batch jobs for global batches 128 and 256. The second job waits for the first to finish, limiting concurrent demand. It prints exact status/summary commands for both. Wait for completion and require Overall PASS with PREFLIGHT ONLY scope. The intentional student NOT RUN line is correct. A submission is not PASS. If it fails, read the printed Slurm log and `assignment_report.json`.

These preflights test one representative personalized seed at both required batch sizes without supplying a student solution. Student full-mode verification still requires the student's own correction; no complete reference solution is published.

## Validation boundary

- Existing guided notebook: actual six-check SciTech PASS, job 24983, haskell, PyTorch 2.8.0+cu128, N=2048. Evidence in `sessions/M4S5_async_distributed_ml/validation/scitech_24983.json`.
- New assignment: source extraction and checked transformations, Python syntax and both generated Slurm script syntax verified locally.
- Local N=4096, personalized seed 3068: C fixture, prepared data, independent candidate workflow, actual checkpoint interruption/restart, and inference passed with real PyTorch. See `validation/local_workflow_report.json`; Overall remains SKIPPED because Gloo cannot initialize in the authoring environment.
- Independent numerical tests: naive unequal local-mean averaging differs from the full-batch update; sample-weighted reference agrees; both batch sizes cover exactly the training rows; starter function is intentionally absent. See `validation/local_math_validation.json`.
- NOT RUN: the new assignment-specific real Gloo harness on SciTech, the representative preflight and batch-256 preflight. Instructor must collect this new evidence; historical guided PASS must not be relabelled as assignment-specific PASS.
- NOT RUN: any student implementation, report, Blackboard bank or authorship check.

No validated reference solution to the student exercise is included in this public repository. Explicitly incorrect code is limited to the labelled negative test.
