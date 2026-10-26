# M4S5 - Practice 4: distributed ML and a complete workflow

Part 1 is a guided notebook. Part 2 will be a separate individual cluster assignment; the conceptual quiz has a different purpose. This branch currently delivers **Part 1 only**, ready for instructor validation on SciTech before assignment/slides are finalized.


## SciTech validation: PASS (job 24983)

Instructor-provided executed notebook HTML confirms all six required checks passed on haskell, 25 October 2026, using the shared image and PyTorch 2.8.0+cu128. See `validation/scitech_24983.json` for extracted evidence and its source hash. This supersedes the earlier NOT RUN cluster status below; the local report remains a separate historical record.

The DDP serial-update maximum parameter difference was 2.78e-17; final 1-worker versus 2-worker differences were 1.67e-16. Replica parameter difference was zero. Recovery from step four after interruption at step seven matched uninterrupted training exactly. Median CPU training times were 0.039802 s (one worker) and 0.155763 s (two): a slower two-worker result is expected to be possible for this tiny workload and is not a correctness failure. Candidate selection and batched inference passed too.

No executable code changed when recording this evidence. Re-running the verifier reuses job 24983 for unchanged sources/configuration. A new job is not necessary merely to repeat the same validation.

## Open in JupyterLab

Use File -> Open from URL:

https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m4s5-distributed-ml-2026/sessions/M4S5_async_distributed_ml/M4S5_async_distributed_ml.ipynb

Run all cells in order using Python 3 on the university JupyterHub. No kernel-side NumPy, PyTorch or plotting installation is required. The notebook generates all executable files itself.

## Teaching scope

Official Session 20 is Practice 4, a small distributed-ML/data-parallel experiment. It consolidates M4S3 (HPC and ML) and M4S4 (distributed ML workflows), with an application connection to M4S1. M4S2 is the midterm, not an additional ML topic.

Five investigations use synthetic machine-condition sensor features:

1. Separate training/validation/test data; prepare and reuse a feature cache; distinguish global batches from worker shards.
2. Inspect real C gradient code, including an unequal-shard counterexample; compare a real two-worker DDP update against the serial global-batch reference.
3. Run one-worker/two-worker CPU Gloo DDP, three independent repetitions each, preserving global batch membership at every step; compare full parameter vectors, training medians and an independently measured All-Reduce payload latency.
4. Train three independent learning-rate candidates in a two-process pool, select by validation accuracy, then evaluate the chosen model on held-out test data.
5. Deliberately interrupt a real child after step seven, resume from step four with optimizer/RNG/cursor state, compare to uninterrupted training, and measure real batched inference calls.

Each investigation includes prediction, code, execution evidence, interpretation, a one-factor student experiment and a takeaway. Data are synthetic; this is not an industrial diagnostic benchmark.

## SciTech execution

- Automatic submission is enabled; one CPU Slurm allocation, one node, two cores, 4 GB, eight-minute limit.
- C compiles/runs natively with the system GCC.
- Python ML work runs in the shared image `/data/software/containers/pytorch-2.8.0-cuda12.8.sif`, without copying/downloading the image and without GPU access.
- The image was accessible in prior instructor evidence, but this new workload's PyTorch/Gloo route still needs SciTech validation.
- Conflicting inherited Slurm memory variables are removed before `sbatch`.
- Settings/source fingerprints create student-owned directories under `~/m4s5_runs/`. Unchanged reruns reuse the recorded job. Changed source/configuration gets a new directory, preventing stale measurements from being reused.
- Notebook submission waits at most 45 seconds; refresh later if queued/running. Default submission does not run the experiment on the notebook host.
- Use numerical `report.json`, `SUMMARY.txt` and the matching Slurm job log together. Submission or COMPLETED alone is not PASS.
- The required route uses real CPU processes on one node; no GPU or multi-node performance claims are made. Detailed Apptainer/hybrid programming remains Module 5 material.

## Instructor quick test

From your ordinary SciTech SSH terminal:

```bash
mkdir -p "$HOME/m4s5-validation"
cd "$HOME/m4s5-validation"
curl -fL 'https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m4s5-distributed-ml-2026/instructor_checks/M4S5/verify_cluster.sh' -o verify_cluster.sh
bash verify_cluster.sh
```

The verifier extracts and executes only setup/source-writing cells from the actual student notebook and submits its exact batch script. It prints the job ID, notebook hash, log path and exact `sacct`/`cat` commands. Wait for completion and require every required check plus Overall to be PASS. Keep `report.json` as evidence. On a failed run inspect logs before deleting `submission.json` to request a deliberate retry; never resubmit repeatedly while an existing job is active.

## Validation report

Authoring date: 25 October 2026. See `validation/local_report.json` for actual locally executed measurements.

- PASS: all notebook code cells parse and execute without Slurm, returning NOT RUN and avoiding heavy execution on the notebook host.
- PASS: embedded source files match the tested executable sources; both shell scripts pass `bash -n`.
- PASS: native C compilation and independent known-value gradient fixture.
- PASS: real prepared-data cache round trip and split/shard checks.
- PASS: real candidate fan-out/fan-in and validation-based selection.
- PASS: deliberate child exit 75, durable checkpoint at step four, fresh-child resume and full-parameter comparison against uninterrupted training.
- PASS: batched versus individual inference predictions and actual compute-call timings.
- SKIPPED locally: real Gloo DDP execution. PyTorch was installed only in the authoring test environment; Gloo initialization was attempted but denied by the environment's socket restrictions (`Operation not permitted`). A hostname-rendezvous issue was also removed by explicitly using loopback for this one-node lab. DDP is **not** claimed locally validated.
- NOT RUN: this revision on SciTech, including the actual shared-image PyTorch import, Gloo process group, allocation and resource/runtime feasibility. The cluster path does not enable the validation-only `--skip-ddp` option and must pass every required check.

The local report therefore correctly says Overall SKIPPED, not PASS. Local timings are not SciTech measurements. No student-authorship claim, grading validation, student optimized solution or completed Blackboard bank is supplied.

## Measurement boundaries

Training time excludes process startup/data preparation and includes the training loop plus final synchronization, reduced using the maximum rank time. The baseline is one-worker DDP. An All-Reduce microbenchmark is not an attribution of communication time within actual DDP. Inference timing measures CPU model calls, excluding network, queueing and batching wait time. Do not assume speedup or a particular selected candidate.

Sources: official teaching plan, Session 20; current M4S3/M4S4 course slides; Oscar Diez, *High-Performance Computing: Applications and Frontiers*, distributed ML/deep learning coverage; Yuan Tang, *Distributed Machine Learning Patterns*, workflow and recovery patterns.
