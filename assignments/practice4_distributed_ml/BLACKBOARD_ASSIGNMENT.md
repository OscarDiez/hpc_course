# HPC26_M4S5 - Practice 4: Correct and reproducible distributed machine learning

**Individual cluster assignment | Module 4 | Two assessed components**

## Purpose

A predictive-maintenance team trains a small classifier from machine-condition sensor features. It wants to use several training workers, compare candidate models and recover interrupted work. Your task is to establish that the computation is correct before deciding whether its parallel execution is useful.

You will use real CPU processes, PyTorch DistributedDataParallel (DDP) and Slurm on SciTech. The data are synthetic, inspired by vibration and operating-condition measurements; this is an educational systems experiment, not a validated industrial diagnostic model.

Complete the guided M4S5 notebook first. In this assignment you must implement a missing correction, run your own experiments, interpret your own evidence and submit one consolidated PDF. Re-running the guided notebook unchanged is not sufficient.

## What you will submit

1. **Blackboard evidence questions: 40% of this assignment.** Complete these after running the experiments. Keep your configuration, source code, job logs and results available when answering. These questions are separate from the module's conceptual quiz.
2. **One consolidated PDF: 60% of this assignment.** Include a 4-6-page main report followed by readable technical appendices containing the required evidence. The deadline and submission availability are those shown in Blackboard.

This is individual work. Follow the course rules for assistance and AI use. Declare assistance used, identify your own implementation and be prepared to explain or reproduce any submitted result. Job IDs, hashes and personalized inputs support checking your work; they do not by themselves prove authorship.

## 1. Set up your own cluster workspace

Use the normal SciTech SSH terminal. Prepare and submit jobs from your home directory. Run computation inside Slurm allocations, not directly in the login session.

Replace `YOUR_STUDENT_ID` with your actual student identifier:

```bash
mkdir -p "$HOME/m4s5-download"
cd "$HOME/m4s5-download"
curl -fL 'https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m4s5-distributed-ml-2026/assignments/practice4_distributed_ml/setup_assignment.sh' -o setup_assignment.sh
bash setup_assignment.sh YOUR_STUDENT_ID "$HOME/m4s5_individual_b128"
cd "$HOME/m4s5_individual_b128"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
sbatch run_baseline.sbatch
```

The setup refuses to overwrite an existing directory. If you need a fresh attempt, choose a new directory name. Your seed is **2026 plus the final four numeric digits of your student identifier**; if it contains fewer than four numeric digits, use all its numeric digits. The setup calculates this automatically and saves it in `config.json` and `provenance.json`.

The supplied job uses one CPU node, two CPU cores, 4 GB and an eight-minute limit. C runs natively; Python uses the centrally installed shared PyTorch image. Do not download a private image, install packages or request GPUs for this assignment. Detailed container programming and MPI/OpenMP hybrid implementation are not required here.

Wait for the baseline job to finish. Use its actual job ID in:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat ASSIGNMENT_SUMMARY.txt
```

The baseline must show **Overall: PASS** with scope **BASELINE ONLY**. The student correction should say **NOT RUN** at this point: you have not implemented it yet. Inspect `assignment_report.json`, `report.json` and `slurm-JOB_ID.out` if a required check fails.

Use these fixed baseline settings:

| Setting | Required baseline |
|---|---|
| Training examples | 4,096 |
| Validation examples | 512 |
| Test examples | 512 |
| Prepared features / classes | 16 / 2 |
| Epochs | 6 |
| Global batch | 128 |
| Dataset seed | Your personalized seed |
| Numerical precision | Supplied float64 |
| Training learning rate | Supplied 0.12 |
| Model and initialization | Supplied implementation, unchanged |
| Workers | One or two; one CPU thread per worker |

Keep the supplied harness and reference checks unchanged. Your required implementation belongs in `student.py`. Put any additional measurement/analysis code in a separate file and include it in your appendices.

## 2. Establish the baseline and inspect the workflow

Before looking at the timings, write a short prediction about one-worker versus two-worker performance.

From your own baseline results:

- Explain the data path: synthetic inputs, prepared features, training/validation/test split, model training, selection, evaluation and inference.
- Explain which data are cached and what the recorded dataset hash identifies.
- Show the C gradient fixture's serial, weighted and incorrectly averaged values. Explain why worker agreement alone cannot establish numerical correctness.
- Record the one-step serial-reference difference, replica difference and final full-training parameter differences for balanced DDP.
- Show all three training times for one worker and balanced two-worker DDP. Calculate median time, speedup relative to one-worker DDP and two-worker efficiency. Explain the limits of these tiny one-node CPU measurements.

Do not claim a speedup merely because more workers were requested. A slower but correct parallel run is a valid outcome.

## 3. Implementation challenge: unequal work, equal influence per sample

The supplied exercise divides every global batch between two workers in a **3:1 ratio**. With global batch 128, rank 0 receives 96 examples and rank 1 receives 32. Both still participate in every optimizer step, and their union is the same global batch used by the reference.

Each worker computes the **mean** loss on its local examples. Ordinary DDP then averages the workers' gradients. Giving equal influence to the two local means does not generally give equal influence to every sample when their counts differ.

The baseline deliberately runs this naïve version. Inspect `unequal-naive.json` and `unequal-naive.log`. Show that replicas agree while the serial-reference comparison fails. An expected failure of this deliberately incorrect computation is a successful negative test, not a reason to suppress errors.

Implement `loss_multiplier(local_count, global_count, world_size)` in `student.py`:

- Return the multiplier applied to a rank's local mean loss before `backward()`.
- Derive the correction from sample counts and DDP's averaging behavior.
- Make it work for both equal and unequal positive local counts, rather than hard-coding a value for one rank.
- Preserve the model, data, global batches, initialization, optimizer and update count.

Do not replace DDP with independent models or a serial run. Do not loosen tolerances, change the reference, remove assertions or edit saved results.

After implementing the function:

```bash
cd "$HOME/m4s5_individual_b128"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
sbatch run_full.sbatch
```

Wait for completion, then inspect `ASSIGNMENT_SUMMARY.txt` and `assignment_report.json`. The full check must show PASS for your correction. It compares one corrected update with the serial reference, checks replica agreement, and compares three complete corrected training runs against the corresponding one-worker models.

In your PDF include your derivation, complete `student.py`, the naïve and corrected numerical differences, both ranks' sample counts and the final full-training comparisons. Compare corrected unequal two-worker times with balanced two-worker times and explain why a 3:1 partition can introduce waiting.

## 4. One-factor experiment: change global batch size

Predict the effect of changing global batch 128 to 256 on local batch sizes, optimizer updates, runtime and validation accuracy. Keep examples, epochs, seed, model, learning rate, precision and CPU resources unchanged.

Create a separate directory so the original evidence remains intact:

```bash
cd "$HOME/m4s5-download"
bash setup_assignment.sh YOUR_STUDENT_ID "$HOME/m4s5_individual_b256"
cp "$HOME/m4s5_individual_b128/student.py" "$HOME/m4s5_individual_b256/student.py"
cd "$HOME/m4s5_individual_b256"
python3 - <<'PY'
import json
from pathlib import Path
p = Path('config.json')
c = json.loads(p.read_text())
c['global_batch'] = 256
p.write_text(json.dumps(c, indent=2) + '\n')
PY
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
sbatch run_full.sbatch
```

Require numerical checks to pass for this configuration too. Present all three timings per training configuration, medians, batch sizes, update counts and validation accuracy for both global batch sizes.

Explain the result: changing global batch changes both work per synchronization and the number of optimizer updates. It is not a pure worker-count comparison. Do not use the test score to decide which batch size is better.

## 5. Model selection, recovery and inference

For the baseline global batch 128, inspect the supplied workflow's results from your own full run:

- Compare the three independent learning-rate candidates. Identify the selected candidate and the validation-based selection rule. Explain why these candidate processes do not synchronize gradients with one another.
- Report the selected model's held-out test accuracy. Candidate selection must use validation data; the test score is reported afterwards.
- Show the intentional child interruption, last durable checkpoint, recovered step, replayed work and final comparison with uninterrupted training. Explain why optimizer state and the next-step cursor matter in addition to model weights.
- Compare inference batch 1 and batch 32. Distinguish compute-call latency from examples per second, and identify the real service costs excluded from this measurement.

For the batch-size comparison in Task 4, use validation metrics. Do not repeatedly tune settings against either run's held-out test output. The supplied harness reports a test result automatically; that does not make the test split a selection tool.

## 6. Prepare one reviewable PDF

Your main report should contain:

1. Problem, workflow diagram and predictions.
2. Personalized configuration, software/resource context and correctness evidence.
3. Unequal-shard derivation, implementation and negative/positive tests.
4. Baseline and batch-size comparison tables, with your interpretation.
5. Model selection, recovery and inference observations.
6. Limitations and a concise conclusion supported by your results.

Include technical appendices in the **same PDF**:

- Complete `student.py`; configuration files for both batch sizes; any additional code you wrote.
- Submitted batch scripts, source hashes, shared-image hash and the pinned source revision recorded in `provenance.json`.
- Job IDs, host names and relevant Slurm accounting/log excerpts for each required run.
- Both full-check summaries; the naïve/corrected update checks and full-training differences.
- All individual training times, calculation methods, selected-candidate record and checkpoint/inference evidence.
- Assistance declaration and any infrastructure failures relevant to your conclusions.

Use readable text and tables. Screenshots may supplement evidence, but must not replace code, raw values or explanations. Do not submit only links, a ZIP or an exported notebook in place of the consolidated PDF.

Suggested filename: `HPC26_M4S5_StudentID.pdf`.

## Assessment of the PDF: 60 points

| Criterion | Points |
|---|---:|
| Reproducible setup, workflow and baseline correctness | 10 |
| Unequal-shard derivation, implementation and numerical evidence | 20 |
| Controlled timing and batch-size comparison | 12 |
| Model selection, checkpoint recovery and inference reasoning | 10 |
| Traceable evidence, clear reporting and limitations | 8 |
| **Total** | **60** |

The remaining 40% comes from the Blackboard evidence questions. A faster result does not automatically earn a higher grade; correctness, controlled comparison and defensible interpretation matter.

## Execution status and infrastructure problems

Use the labels accurately: **PASS**, **FAIL**, **SKIPPED**, **NOT RUN**. Submitted, pending or running jobs have not passed yet. The baseline's student section is intentionally NOT RUN; the final student correction must actually execute and pass. Never present an expected negative-test failure as a successful corrected implementation.

If the shared image, compiler, Slurm allocation or distributed runtime is unavailable, retain the job ID, exact error and logs, and contact the lecturer. Do not silently replace required cluster evidence with invented measurements or a claim of multi-node/GPU execution. Continue with work that does not depend on the unavailable component and identify what remains untested.
