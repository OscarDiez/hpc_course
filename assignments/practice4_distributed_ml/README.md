# Practice 4 - Individual distributed ML assignment

Read [the Blackboard assignment brief](BLACKBOARD_ASSIGNMENT.md). Submit the assignment evidence questions and one consolidated PDF; the module quiz has a separate purpose.

## Student setup

Run `setup_assignment.sh YOUR_STUDENT_ID NEW_DIRECTORY` from SciTech SSH. The setup refuses to overwrite an existing directory, derives your personalized seed and prepares the supplied teaching sources. It downloads no packages or container images.

The generated files are:

- `student.py`: your required loss-multiplier implementation; initially unfinished.
- `ml_lab.py`: supplied model, data, DDP, workflow, recovery and inference code.
- `gradient.c`: numerical gradient example.
- `run_experiments.py`: student baseline and implementation correctness checks.
- `config.json` and `provenance.json`: your settings and source identity.
- `run_baseline.sbatch`: run the supplied baseline experiments and intentional incorrect unequal-shard example.
- `run_full.sbatch`: run the baseline plus your completed correction and full-training comparisons.

The setup helpers extract sources from the pinned guided notebook. Your own experiment directories contain only student activities and their supporting files.

## Student workflow

1. Submit `run_baseline.sbatch`, wait for completion and read the matching `ASSIGNMENT_SUMMARY.txt`, `assignment_report.json` and Slurm log. Baseline PASS does not validate an unfinished student function.
2. Implement `loss_multiplier()` in `student.py` without altering the supplied reference checks.
3. Submit `run_full.sbatch` and require the numerical comparisons to pass.
4. Repeat the required global-batch experiment in a separate directory, following the brief.
5. Preserve raw timings, numerical checks, model-selection/recovery/inference evidence and the matching job IDs for your PDF and Blackboard questions.

The two-rank unequal case assigns three quarters of each global batch to rank 0 and one quarter to rank 1. The combined samples must have the same influence as in the serial global-batch reference. The deliberately wrong local-mean average must be detected; it is not a valid student correction.

Use one job per working directory at a time. Inspect output only for the matching completed job. Earlier generated outputs are preserved under `previous_outputs/`; separate batch-size directories prevent one experiment from replacing another.

## Resources and results

Use the supplied CPU Slurm scripts: one node, two cores, 4 GB and an eight-minute limit. The centrally installed image is `/data/software/containers/pytorch-2.8.0-cuda12.8.sif`. C runs natively; Python ML work uses the image. No GPU is required.

PASS means the executed numerical checks passed. An intentional naïve result failure is a successful negative test when it is correctly detected. A submitted, pending or running job has no final success result. An unfinished student function remains NOT RUN in the baseline and must actually pass in the full run.

The baseline uses real one-node CPU Gloo DDP; these timings do not establish GPU or multi-node scaling. The supplied measurements exclude process startup/data preparation from training-loop timing. Inference measurements are model-call timings, excluding network and queueing delay. A slower correct parallel run remains a valid observation.
