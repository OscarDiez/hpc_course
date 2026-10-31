# M6S1 / Session 26 — Neuromorphic computing: updated slide plan

**IE University · BCSAI · High-Performance Computing · 2026–27**  
**Updated 31 October 2026. PowerPoint has not been modified.**

This replaces the earlier broad alternative-paradigms plan for this lesson. Follow the revised teaching scope requested: **one simple neuromorphic notebook for one session**. Quantum, photonic and in-memory demonstrations from the earlier plan are not part of this live sequence.

## Baseline and scope

Use **HPC_S26_Quantum_Neuromorphic_Alternative_2026-27.pptx**, the original **108-slide** deck whose cover says “6.2 Introduction to Neuromorphic computing”. Despite its filename, its contents are neuromorphic. Save a working copy as **HPC_M6S1_Neuromorphic_2026-27.pptx** when editing is requested. All source numbers below refer to the untouched original, before any moves, not to positions after reordering.

The official syllabus Session 26 has a wider scope: quantum, neuromorphic, photonic and in-memory computing, their limitations and hybrid HPC workflows. **This neuromorphic-only lesson is the user-requested subdivision, not full coverage of that syllabus session.** The remaining paradigms still require their own teaching/notebook coverage in the Module 6 plan. Do not claim that this notebook satisfies all Session 26 topics. Official Session 27 remains exascale/post-exascale systems; its timetable has not been changed here.

**Central goal:** students explain state, leak, threshold, spike and reset; read two short C programs; change one setting; interpret output; distinguish a concept demonstration from hardware evidence.

**Course continuity:** use familiar C, AI model concepts and reproducible workflows from earlier sessions. Preserve the original IE identity and painting layouts. Avoid framework installation, training pipelines, brain-equivalence claims and hardware specification catalogues.

**What changes from the source:** merge repeated definitions; rebuild misleading ANN/SNN and architecture tables; replace long or incomplete framework code with the verified C model; replace old quizzes with five distributed reasoning questions; keep advanced material in Backup.

## A. Notebook that drives these slides

- [Focused neuromorphic notebook](https://github.com/OscarDiez/hpc_course/blob/teaching/m6s1-alternative-paradigms-2026/sessions/M6S1_neuromorphic/M6S1_neuromorphic.ipynb)
- [Raw notebook — JupyterLab File → Open from URL](https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m6s1-alternative-paradigms-2026/sessions/M6S1_neuromorphic/M6S1_neuromorphic.ipynb)
- [Validation report](https://github.com/OscarDiez/hpc_course/blob/teaching/m6s1-alternative-paradigms-2026/sessions/M6S1_neuromorphic/VALIDATION.md)
- [Pull request #54](https://github.com/OscarDiez/hpc_course/pull/54)

| Notebook section | Actual student task | Verified local evidence | Slide connection |
|---|---|---|---|
| 0. Open and run | Use Python 3; inspect C; compile/run tiny examples | Runner and all eight code cells executed locally | Setup before class, slides 1–2 |
| 1. Make a neuron spike | Threshold 1.0 → 1.5, retention fixed at 0.8 | Four → two spikes in eight steps | 5–10 |
| 2. What does the leak do? | Hold threshold 1.0; change retention | Retention 0.0/0.5/0.9 gives 0/2/4 spikes | 11–13 |
| 3. Send changes rather than repeated readings | Reconstruct a sensor stream; replace its input array | Baseline eight samples/three messages; alternating stream eight/eight | 14–19 |
| 4. Explain what you learned | Four short explanations | Expected reasoning supplied; no auto-graded assessment | 20–24, 26–27 |
| 5. Optional external practice | Guided snnTorch Colab or EPFL extension | External exercises NOT RUN by the author | 25 |

**Validation boundary:** saved outputs are LOCAL_CPU. Eight code cells executed in order with a Python harness; both C programs compile under GCC 13.3.0 with warnings treated as errors. Five neuron parameter settings match an independent recurrence; both sensor reconstructions pass. Actual SciTech execution, JupyterLab URL import and external practice execution are **NOT RUN**. Neuromorphic hardware execution is **SKIPPED**. No speed/energy benchmark was performed.

**Practical classroom setup:** open and save a personal notebook before the lecture. The runner creates student-owned directories, uses GCC and invokes a one-task `srun` when Slurm is present (cpu partition, one CPU, 1 GB, two-minute limit). Missing GCC reports SKIPPED; execution/compilation errors report FAIL. Do not narrate the runner’s plumbing live. If a job queues or fails, show the labelled saved local results and identify the missing cluster validation. A queued job is not a result. No package installation is required for the core notebook.

## B. Move to Backup first

Keep an untouched original. Create Backup after the live teaching sequence. Groups are disjoint; all 108 original slides are accounted for by reuse, Backup or delete candidates. Backup preserves material; it does not certify its claims. Mark unverified material “REFERENCE ONLY — REQUIRES CORRECTION”.

| Group | Exact original slide numbers | Reason / handling |
|---|---|---|
| Duplicates and overly detailed explanations | **8, 16, 38, 42, 52, 53, 77, 78, 81, 89, 90, 101, 102** | Merged/replaced by compact model and comparison slides. Correct old ANN/static-input claims before any reuse. |
| Advanced programming and reference | **29, 34, 47, 48, 82, 83, 85, 86, 87, 93, 94, 95, 96, 97, 98, 103, 104, 106, 107, 108** | Frameworks, training, neuron taxonomy and hardware detail are outside the simple practical route. |
| Other session topics and advanced device detail | **36, 37, 88, 100** | Keep memristors/crossbar detail for a device-focused extension; conventional DNN acceleration belongs with GPU/AI teaching. Do not imply memristors are required. |
| Optional history and case studies | **9, 12, 17, 24, 26, 27, 28, 44, 45, 79, 84, 91** | Preserve source material without spending live time on dated device tables, history or speculative applications. Slide 26 needs technical correction before reuse. |
| Old questions and answer slides | **4, 19, 20, 32, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75** | Replace with five reasoning checkpoints distributed through this plan; old questions require independent review. |
| Other supplementary content | **5, 10, 39, 56, 76, 80, 99** | Spare dividers, unverified brain-efficiency facts and philosophical/AGI material. Use original 76 as Backup divider. |

**DELETE candidates after preserving the untouched source: 33, 49, 57, 92.** Slide 33 mixes framework names and unsupported execution claims; 49 is incomplete/mislabelled classifier code with random output; 57 contains misleading learning/deployment advice; 92 presents a dated device list as current. Deletion is optional; quarantining them in Backup is acceptable.

**Sources reused or merged:** 1, 2, 3, 6, 7, 11, 13, 14, 15, 18, 21, 22, 23, 25, 30, 31, 35, 40, 41, 43, 46, 50, 51, 54, 55, 58, 59, 60, 61, 105. A source used in a merge is not an extra live slide: retain its unused details only in the untouched original or a clearly labelled supplementary copy.

## C. Final teaching order

**27 slides · 80 planned minutes**, including five MCQs, predictions and demonstrations. Timing is a teaching budget; queue delays and optional external practice are outside it.

| New # | Current source | Final title | Action | Notebook connection | Infographic / visual | Minutes |
|---|---|---|---|---|---|---:|
| 1 | 1 | M6S1 — Neuromorphic computing | UPDATE | Introduction | Reuse layout / code / table | 1 |
| 2 | 2 | What will you be able to explain? | HEAVY UPDATE | Sections 1–3 | Reuse layout / code / table | 2 |
| 3 | 3 | [MCQ · PREDICT] Which sensor is a promising candidate? | HEAVY UPDATE | Section 3 problem, before running | Reuse layout / code / table | 3 |
| 4 | 6 + 7 + 11 | Why investigate a different architecture? | MERGE | Section 3 motivation | ADD INFOGRAPHIC | 2 |
| 5 | 13 + 14 + 15 + 21 | What is neuromorphic computing? | MERGE | Section 1 introduction | ADD INFOGRAPHIC | 3 |
| 6 | 22 + 105 | A neuron stores evidence over time | MERGE | Section 1, neuron.c variables | ADD INFOGRAPHIC | 4 |
| 7 | NEW | [PREDICT] Will the first input trigger a spike? | ADD | Section 1 prediction | Reuse layout / code / table | 3 |
| 8 | NEW | Read the three lines that matter | ADD | Section 1, neuron_c; full eight-step loop in notebook | Reuse layout / code / table | 3 |
| 9 | NEW | [DEMO] A higher threshold changes the spike train | ADD | Section 1 baseline and higher_threshold run cells | ADD INFOGRAPHIC | 5 |
| 10 | 31 | [MCQ] Did the zero state mean nothing happened? | HEAVY UPDATE | Section 1 output columns | Reuse layout / code / table | 3 |
| 11 | NEW | Leak means old evidence fades | ADD | Section 2, same neuron.c | ADD INFOGRAPHIC | 3 |
| 12 | NEW | [ACTIVITY] Change how much the neuron remembers | ADD | Section 2 no_memory and student_result cells | Reuse layout / code / table | 5 |
| 13 | NEW | [MCQ] Why does zero retention prevent spikes here? | ADD | Section 2 zero-retention result | Reuse layout / code / table | 3 |
| 14 | 30 + 41 | An event can say “something changed” | MERGE | Section 3 problem and samples[] | ADD INFOGRAPHIC | 4 |
| 15 | NEW | [PREDICT] How many messages do we need? | ADD | Section 3 prediction | Reuse layout / code / table | 2 |
| 16 | NEW | Read the change-detection loop | ADD | Section 3 events_c loop | Reuse layout / code / table | 3 |
| 17 | NEW | [DEMO] Eight readings, three messages | ADD | Section 3 events run cell | Reuse layout / code / table | 4 |
| 18 | NEW | [ACTIVITY] What if every reading changes? | ADD | Section 3 student experiment; edit events_c array | Reuse layout / code / table | 4 |
| 19 | NEW | [MCQ] What did three messages actually prove? | ADD | Section 3 result and evidence boundary | Reuse layout / code / table | 3 |
| 20 | 18 + 40 + 43 | [MCQ] Can a non-spiking network process time? | MERGE | Sections 1–2 model interpretation | Reuse layout / code / table | 3 |
| 21 | 23 + 25 | A CPU model is not a neuromorphic chip | MERGE | Notebook validation and limits | ADD INFOGRAPHIC | 3 |
| 22 | 46 + 50 | Learning is a separate step | MERGE | Section 1 fixed-input model; Section 4 explanations | Reuse layout / code / table | 2 |
| 23 | 35 | Where could this fit beside HPC? | HEAVY UPDATE | Sections 3–4 architectural interpretation | ADD INFOGRAPHIC | 3 |
| 24 | 51 | [DISCUSS] Would you investigate a specialist? | HEAVY UPDATE | Section 4 question 4 | Reuse layout / code / table | 4 |
| 25 | 58 + 59 + 60 | Optional practice: explore a visual neuron | MERGE | Section 5 optional external practice | Reuse layout / code / table | 2 |
| 26 | 54 + 55 | [TAKEAWAY] 4 essentials from Session 26 | MERGE | Sections 1–4 consolidation | Reuse layout / code / table | 2 |
| 27 | 61 | Exit question: what would you measure next? | UPDATE | Section 4 written answers; Session 27 bridge | Reuse layout / code / table | 1 |

## D. Visual identity

Preserve IE logos, source master, 16:9 proportions, title treatment and painting-style question/discussion layouts. Keep painting proportions and existing attribution. Technical diagrams use white backgrounds, large sans-serif labels, rectangular elements and clear arrows: blue `#517C9C`, grey `#777777`, muted green `#668873`. Aim for 22–24 pt body text; rebuild dense tables rather than shrinking them. Keep slide code to 3–4 meaningful lines; the complete programs stay in the notebook.

Use native/editable shapes for exact model relationships and tables/charts for counts. Never use an invented trace. Spike rasters must match the executed C outputs. MCQ answers and distractor explanations appear only in teacher notes. All demonstration slides label evidence as CPU simulation/counts, with LOCAL_CPU on saved outputs until replaced by recorded SciTech results.

## E. Complete slide specifications

### Slide 1 — M6S1 — Neuromorphic computing

**Slide number and source:** New 1; original 1.  
**Action:** UPDATE.  
**Title:** M6S1 — Neuromorphic computing  
**Subtitle:** From sensor inputs to spikes: read, predict, run and explain.

**Student-visible content**

- High-Performance Computing · BCSAI · IE University
- 2026–27 · Oscar Diez
- Today: one neuron, retained state and sensor events.

**Infographic / visual:** Preserve original cover painting, IE logo and title placement; replace 6.2 with M6S1. No new device photograph needed.

**Bottom banner / takeaway:** Can a computer respond mainly to changes?

**Notebook connection:** Introduction.

**Teacher notes — 1 minutes**

- Introduce the revised neuromorphic-only scope.
- Connect to earlier AI and heterogeneous computing sessions.
- Ask for an example of a sensor that usually repeats its reading.
- Expected: a room-temperature or stationary-motion sensor.
- Make clear that today’s code runs on a CPU; a chip is not being accessed.

### Slide 2 — What will you be able to explain?

**Slide number and source:** New 2; original 2.  
**Action:** HEAVY UPDATE.  
**Title:** What will you be able to explain?  
**Subtitle:** Three small experiments reveal the main ideas.

**Student-visible content**

- Trace a neuron’s state, threshold, spike and reset.
- Predict what happens when threshold or retention changes.
- Count messages when a sensor sends only changes.
- Explain what CPU simulations cannot tell us about hardware.

**Infographic / visual:** Use existing agenda layout with three large numbered experiment blocks and a small evidence label.

**Bottom banner / takeaway:** Understand the mechanism before comparing processors.

**Notebook connection:** Sections 1–3.

**Teacher notes — 2 minutes**

- Show the simple notebook, not the old broad notebook.
- Introduce the three tasks without new frameworks.
- Ask students to keep predictions beside observed results.
- Explain that threshold and retention are different parameters.
- Tell students the optional external tutorial can be completed after class.

### Slide 3 — [MCQ · PREDICT] Which sensor is a promising candidate?

**Slide number and source:** New 3; original 3.  
**Action:** HEAVY UPDATE.  
**Title:** [MCQ · PREDICT] Which sensor is a promising candidate?  
**Subtitle:** A battery-powered monitor must respond quickly to changes.

**Student-visible content**

- Which observation gives the strongest reason to investigate event communication?
- A. Most readings repeat, with occasional meaningful changes.
- B. Every reading changes and must always be transmitted.
- C. The device has a GPU, so no comparison is needed.
- D. Any brain-inspired design uses no idle power.

**Infographic / visual:** Preserve original scenario/question identity; shorten the scenario. Four equally styled answer lines; no answer cues.

**Bottom banner / takeaway:** OMIT

**Notebook connection:** Section 3 problem, before running.

**Teacher notes — 3 minutes**

- Correct answer: A; sparse changes provide a reason to investigate.
- Ask what measurement would still be needed before choosing hardware.
- Expected: latency, detection quality and whole-system energy.
- B provides no message sparsity; C avoids the workload comparison.
- D is an unsupported absolute; event-driven circuitry can still consume idle power.

### Slide 4 — Why investigate a different architecture?

**Slide number and source:** New 4; original 6 + 7 + 11.  
**Action:** MERGE.  
**Title:** Why investigate a different architecture?  
**Subtitle:** Moving and repeatedly processing data can be expensive.

**Student-visible content**

- Processors already exploit several forms of parallelism.
- Repeated reads and data movement also consume resources.
- Some workloads contain long quiet periods.
- A specialised design must still meet accuracy and response requirements.

**Infographic / visual:** [ADD INFOGRAPHIC: grey Sensor → blue Processing → grey Memory blocks; highlight repeated transfers with grey loop arrows and a second muted-green change-event path. Label the paths Repeated samples and Changes. No efficiency numbers.]

**Bottom banner / takeaway:** Useful work and communication both matter.

**Notebook connection:** Section 3 motivation.

**Teacher notes — 2 minutes**

- Retain only the energy/data-movement lesson from the source slides.
- Avoid a long history of scaling or re-teaching Amdahl’s law.
- Ask whether removing a message means the sensor consumed no energy.
- Expected: no; sensing, detection and idle activity still have costs.
- Do not describe conventional processors as inherently sequential.

### Slide 5 — What is neuromorphic computing?

**Slide number and source:** New 5; original 13 + 14 + 15 + 21.  
**Action:** MERGE.  
**Title:** What is neuromorphic computing?  
**Subtitle:** Computing architectures can borrow ideas from nervous systems.

**Student-visible content**

- Neurons maintain state and produce outputs.
- Synapses connect neurons and weight inputs.
- Many spiking designs communicate using events.
- Hardware, model and software must work together.

**Infographic / visual:** [ADD INFOGRAPHIC: three blue neuron rectangles with small grey state boxes, connected by directional arrows labelled weighted input / spike. Use muted green only for a firing output. Keep text large.]

**Bottom banner / takeaway:** Brain-inspired does not mean a complete brain simulation.

**Notebook connection:** Section 1 introduction.

**Teacher notes — 3 minutes**

- Explain the term without suggesting consciousness or human intelligence.
- Use one input, one neuron and one output before discussing a network.
- Ask whether all neurons must learn during inference.
- Expected: no; weights may remain fixed.
- Clarify that today focuses on a spiking model; neuromorphic approaches are broader.

### Slide 6 — A neuron stores evidence over time

**Slide number and source:** New 6; original 22 + 105.  
**Action:** MERGE.  
**Title:** A neuron stores evidence over time  
**Subtitle:** State, threshold and reset make a spike understandable.

**Student-visible content**

- State: evidence retained from previous steps.
- Input: new weighted evidence, 0.6 per step here.
- Threshold: the value needed to emit a spike.
- Reset: set state to zero after firing in our model.

**Infographic / visual:** [ADD INFOGRAPHIC: blue Input 0.6 → blue State box → diamond-style threshold decision represented with a rectangular Test box → green Spike. Return grey Reset to 0 arrow to State. Show retention on the previous-state loop.]

**Bottom banner / takeaway:** State is a value; a spike is an event.

**Notebook connection:** Section 1, neuron.c variables.

**Teacher notes — 4 minutes**

- Point to each matching C variable in the notebook.
- State that numbers are dimensionless teaching values.
- Ask whether the state must be zero when no spike occurs.
- Expected: no; subthreshold evidence can remain.
- Reset to zero is this model’s convention, not the only possible reset rule.

### Slide 7 — [PREDICT] Will the first input trigger a spike?

**Slide number and source:** New 7; original NEW.  
**Action:** ADD.  
**Title:** [PREDICT] Will the first input trigger a spike?  
**Subtitle:** Start at zero with threshold 1.0 and retention 0.8.

**Student-visible content**

- Rule: new state = 0.8 × previous state + 0.6.
- Step 1: predict the state and spike.
- Step 2: predict the state before reset.
- Write your predictions before opening the output.

**Infographic / visual:** Two-row blank table: Step / State before reset / Spike / State after reset. Show the rule above; no prefilled answers.

**Bottom banner / takeaway:** Predict first; use the table to check.

**Notebook connection:** Section 1 prediction.

**Teacher notes — 3 minutes**

- Give students a minute to trace two steps.
- Expected step 1: 0.6, no spike, state remains 0.6.
- Expected step 2: 1.08, spike, reset to zero.
- Common mistake: adding input before applying retention gives a different rule.
- Do not reveal the output until students commit to predictions.

### Slide 8 — Read the three lines that matter

**Slide number and source:** New 8; original NEW.  
**Action:** ADD.  
**Title:** Read the three lines that matter  
**Subtitle:** The C program follows the same rule at each step.

**Student-visible content**

- state = retention * state + input;
- int spike = state >= threshold;
- if (spike) state = 0.0;
- The notebook prints state before and after reset.

**Infographic / visual:** Large three-line code panel, with separate short labels Integrate / Test / Reset. Omit helper and compiler plumbing.

**Bottom banner / takeaway:** The code is the model.

**Notebook connection:** Section 1, neuron_c; full eight-step loop in notebook.

**Teacher notes — 3 minutes**

- Read the recurrence aloud in ordinary language.
- Explain that >= includes the exact threshold.
- Point out that before_reset is recorded before any reset.
- Ask whether this code trains weights; expected answer: no.
- Use the full notebook to locate the loop, but keep only these lines on the slide.

### Slide 9 — [DEMO] A higher threshold changes the spike train

**Slide number and source:** New 9; original NEW.  
**Action:** ADD.  
**Title:** [DEMO] A higher threshold changes the spike train  
**Subtitle:** Run the same C model with one changed parameter.

**Student-visible content**

- Baseline: run_c("neuron", 1.0, 0.8).
- Then: run_c("neuron", 1.5, 0.8).
- Compare four spikes with two over eight steps.
- Keep input and retention unchanged.

**Infographic / visual:** [ADD INFOGRAPHIC: two aligned eight-step spike rasters. Threshold 1.0: steps 2,4,6,8. Threshold 1.5: steps 4,8. Green tick marks, grey time axis; caption LOCAL_CPU saved C simulation, not chip measurements.]

**Bottom banner / takeaway:** A higher threshold requires more accumulated evidence here.

**Notebook connection:** Section 1 baseline and higher_threshold run cells.

**Teacher notes — 5 minutes**

- Run both cells; show printed tables before the summary diagram.
- Actual saved local counts are four and two.
- Ask why step 3 at threshold 1.5 still does not fire: 1.464 < 1.5.
- Explain that PASS checks the expected model result, not useful detection accuracy.
- If SciTech is queued, use labelled saved local output and identify its origin.

### Slide 10 — [MCQ] Did the zero state mean nothing happened?

**Slide number and source:** New 10; original 31.  
**Action:** HEAVY UPDATE.  
**Title:** [MCQ] Did the zero state mean nothing happened?  
**Subtitle:** At step 2, before_reset is 1.08, spike is 1, after_reset is 0.

**Student-visible content**

- Which explanation matches the code?
- A. Input was absent throughout step 2.
- B. The state crossed threshold, emitted a spike, then reset.
- C. The CPU failed because the two state columns differ.
- D. The model learned a zero synaptic weight.

**Infographic / visual:** One output row followed by A–D. No green highlight on the correct answer. Replace the original ANN/SNN question entirely.

**Bottom banner / takeaway:** OMIT

**Notebook connection:** Section 1 output columns.

**Teacher notes — 3 minutes**

- Correct answer: B; the columns refer to different moments in one step.
- Ask students to identify the reset statement.
- A ignores the positive input and before-reset state.
- C mistakes model semantics for an error.
- D invents learning; the program has fixed input and no weight updates.

### Slide 11 — Leak means old evidence fades

**Slide number and source:** New 11; original NEW.  
**Action:** ADD.  
**Title:** Leak means old evidence fades  
**Subtitle:** Retention is the fraction of previous state that survives.

**Student-visible content**

- Retention 0.8 keeps 80% of the previous state.
- Retention 0.0 keeps none of the previous state.
- A smaller retention means a stronger leak.
- Threshold remains 1.0 in the next experiment.

**Infographic / visual:** [ADD INFOGRAPHIC: previous state 1.0 in grey → Retain ×0.8 blue block → 0.8, then +0.6 Input → 1.4 before threshold test. Explicitly label this as an illustrative step, not a row from the baseline trace.]

**Bottom banner / takeaway:** Retention describes what is kept.

**Notebook connection:** Section 2, same neuron.c.

**Teacher notes — 3 minutes**

- Distinguish retention from a leak rate.
- Explain that input is added after the old state is reduced.
- Ask if retention zero also removes new input; expected answer: no.
- Do not introduce differential equations or biological units.
- Return to the exact recurrence to settle ordering questions.

### Slide 12 — [ACTIVITY] Change how much the neuron remembers

**Slide number and source:** New 12; original NEW.  
**Action:** ADD.  
**Title:** [ACTIVITY] Change how much the neuron remembers  
**Subtitle:** Change only retention and record the spike count.

**Student-visible content**

- Run threshold 1.0 with retention 0.0.
- Use the student cell with retention 0.5, then 0.9.
- Record each count and explain the difference.
- Leave input 0.6 and eight steps unchanged.

**Infographic / visual:** Large table with Retention / Prediction / Observed spikes. Rows 0.0,0.5,0.9; observations initially blank on student slide.

**Bottom banner / takeaway:** Change one parameter so the result is interpretable.

**Notebook connection:** Section 2 no_memory and student_result cells.

**Teacher notes — 5 minutes**

- Local verified counts: 0, 2 and 4 for retention 0.0, 0.5 and 0.9.
- Ask students to predict before changing the cell.
- For zero retention, each before-reset value is 0.6.
- For 0.5, spikes occur at steps 3 and 6; for 0.9, at 2,4,6,8.
- Spend the final minute on an explanation rather than changing more parameters.

### Slide 13 — [MCQ] Why does zero retention prevent spikes here?

**Slide number and source:** New 13; original NEW.  
**Action:** ADD.  
**Title:** [MCQ] Why does zero retention prevent spikes here?  
**Subtitle:** The input is 0.6, the threshold is 1.0 and retention is zero.

**Student-visible content**

- A. Each new state is 0.6, so it stays below threshold.
- B. Retention zero changes the input to zero.
- C. Any input must cause at least one spike.
- D. All spiking models require a GPU.

**Infographic / visual:** Use the same MCQ layout as slide 10 and a single reminder of the recurrence.

**Bottom banner / takeaway:** OMIT

**Notebook connection:** Section 2 zero-retention result.

**Teacher notes — 3 minutes**

- Correct answer: A; prior state contributes nothing, new input still contributes 0.6.
- Invite students to compute one step.
- B confuses the input with retained state.
- C ignores the threshold rule.
- D confuses the model with its execution platform; our C model ran on CPU.

### Slide 14 — An event can say “something changed”

**Slide number and source:** New 14; original 30 + 41.  
**Action:** MERGE.  
**Title:** An event can say “something changed”  
**Subtitle:** Send a useful change instead of repeatedly sending the same value.

**Student-visible content**

- Sensor readings: 20,20,20,21,21,21,19,19.
- Send the first value so the receiver starts correctly.
- Send another value only when it changes.
- The receiver retains the last value between messages.

**Infographic / visual:** [ADD INFOGRAPHIC: aligned Sample and Message rows across t=0…7. Samples are eight grey boxes; messages at t=0,3,6 are blue boxes. Grey retention line connects receiver values between events. Label units arbitrary sensor readings.]

**Bottom banner / takeaway:** Events can reduce communication when changes are sparse.

**Notebook connection:** Section 3 problem and samples[].

**Teacher notes — 4 minutes**

- Explain the event-coding analogy without calling it a complete SNN.
- Distinguish a change-message threshold from the neuron firing threshold.
- Ask why the first value must be sent.
- Expected: otherwise the receiver cannot reconstruct its initial reading.
- Note that this example sends exact integer changes and ignores transport loss.

### Slide 15 — [PREDICT] How many messages do we need?

**Slide number and source:** New 15; original NEW.  
**Action:** ADD.  
**Title:** [PREDICT] How many messages do we need?  
**Subtitle:** Reconstruct every sample using the initial value and later changes.

**Student-visible content**

- Input: 20,20,20,21,21,21,19,19.
- Mark each position where a message is sent.
- Count the initial message.
- Predict whether any reading is lost.

**Infographic / visual:** Eight sample boxes with an empty message row for annotation. Avoid showing the three-event answer.

**Bottom banner / takeaway:** OMIT

**Notebook connection:** Section 3 prediction.

**Teacher notes — 2 minutes**

- Expected message positions: t=0,3,6.
- Expected total: three, not two.
- Ask what the receiver outputs at t=4.
- Expected: it keeps 21.
- Clarify zero-based sensor time labels versus one-based neuron step labels.

### Slide 16 — Read the change-detection loop

**Slide number and source:** New 16; original NEW.  
**Action:** ADD.  
**Title:** Read the change-detection loop  
**Subtitle:** Update the receiver only when the incoming value differs.

**Student-visible content**

- if (samples[t] != last) {
-     last = samples[t];
-     messages++;
- }

**Infographic / visual:** Four-line C panel; adjacent notes say last = receiver value and initial message counted before loop. No additional pseudocode.

**Bottom banner / takeaway:** An unchanged value reuses the previous receiver state.

**Notebook connection:** Section 3 events_c loop.

**Teacher notes — 3 minutes**

- Point to the initialisation before the loop.
- Explain that != compares readings, not floating-point noise in this integer example.
- Ask which variable represents receiver memory; expected: last.
- The complete code checks reconstruction for every sample.
- Important limitation: this CPU loop still inspects every input sample.

### Slide 17 — [DEMO] Eight readings, three messages

**Slide number and source:** New 17; original NEW.  
**Action:** ADD.  
**Title:** [DEMO] Eight readings, three messages  
**Subtitle:** The receiver reconstructs the complete noiseless sequence.

**Student-visible content**

- Run the events.c example.
- Output: SAMPLES=8 MESSAGES=3.
- The reconstruction check reports PASS.
- These are message counts, not energy measurements.

**Infographic / visual:** Native two-bar count chart: Samples inspected 8; Messages sent 3. Add Reconstructed samples 8 as text. Caption LOCAL_CPU simulation; counts only.

**Bottom banner / takeaway:** Fewer messages can preserve the information.

**Notebook connection:** Section 3 events run cell.

**Teacher notes — 4 minutes**

- Run the code and show the initial/change messages.
- Verify values 20,21,19 at the three message positions.
- Ask what PASS checks: all reconstructed readings match.
- Do not call this a reduction from eight computations to three: the loop reads all eight.
- Do not convert the message ratio into a speedup or energy claim.

### Slide 18 — [ACTIVITY] What if every reading changes?

**Slide number and source:** New 18; original NEW.  
**Action:** ADD.  
**Title:** [ACTIVITY] What if every reading changes?  
**Subtitle:** Use a second input stream to test the limit of event communication.

**Student-visible content**

- Replace the C array with 20,21,20,21,20,21,20,21.
- Predict the message count.
- Rerun the source-writing cell, then the run cell.
- Explain the old assertion failure before updating the expected count to eight.

**Infographic / visual:** Two aligned timelines: Quiet stream versus Alternating stream. Student message totals stay blank until discussion.

**Bottom banner / takeaway:** Sparsity is a workload property.

**Notebook connection:** Section 3 student experiment; edit events_c array.

**Teacher notes — 4 minutes**

- Verified alternating-stream count is eight; reconstruction still passes.
- The run cell initially checks the original three-message result and intentionally fails after this edit.
- Explain that this is an outdated expected value, not necessarily an incorrect program.
- Only update the assertion after students explain the changed input.
- Restore the baseline array and its assertion before saving the default demonstration copy.

### Slide 19 — [MCQ] What did three messages actually prove?

**Slide number and source:** New 19; original NEW.  
**Action:** ADD.  
**Title:** [MCQ] What did three messages actually prove?  
**Subtitle:** Our C program reconstructed eight readings with three messages.

**Student-visible content**

- A. Every neuromorphic chip uses three eighths of the energy.
- B. The CPU examined only three samples.
- C. This input needs three messages in this exact encoding; hardware savings require measurements.
- D. Communication is free whenever a program uses events.

**Infographic / visual:** Question layout with the single observed count line above A–D.

**Bottom banner / takeaway:** OMIT

**Notebook connection:** Section 3 result and evidence boundary.

**Teacher notes — 3 minutes**

- Correct answer: C; scope of the evidence is input, encoding and reconstruction.
- A invents a device-energy measurement.
- B contradicts the loop that reads every sample.
- D ignores interface, routing and idle costs.
- Ask for a fair follow-up: compare required quality, latency and complete-system energy.

### Slide 20 — [MCQ] Can a non-spiking network process time?

**Slide number and source:** New 20; originals 18 + 40 + 43.  
**Action:** MERGE.  
**Title:** [MCQ] Can a non-spiking network process time?  
**Subtitle:** Our spiking model retains state between inputs.

**Student-visible content**

- Which interpretation is correct?
- A. Only a spiking neural network can retain state or process sequences.
- B. Spiking networks belong outside the category of neural networks.
- C. Both spiking and some non-spiking networks can process time; our model explicitly uses state and spikes.
- D. All non-spiking networks must execute sequentially on a CPU.

**Infographic / visual:** Use the same clean MCQ layout. Small reminder: state → threshold → spike. Remove the old ANN = static and CPU = sequential table; do not show its false categories.

**Bottom banner / takeaway:** OMIT on the question slide.

**Notebook connection:** Sections 1–2 model interpretation; Section 4 question 2.

**Teacher notes — 3 minutes**

- Correct answer: C; state and time are not exclusive to spiking models.
- A ignores non-spiking recurrent networks and other sequence-processing architectures.
- B is a category error: an SNN is a neural-network model.
- D falsely fixes both execution hardware and parallelism from the model label.
- After the vote, say: “Spiking describes a model feature; it does not automatically identify the best hardware.”

### Slide 21 — A CPU model is not a neuromorphic chip

**Slide number and source:** New 21; original 23 + 25.  
**Action:** MERGE.  
**Title:** A CPU model is not a neuromorphic chip  
**Subtitle:** The same model can be mapped to different implementations.

**Student-visible content**

- Today: sequential C simulation on a conventional CPU.
- Specialised systems can implement neurons, state and event routing.
- Digital and analogue implementations have different tradeoffs.
- A memristor is one possible device option, not a requirement.

**Infographic / visual:** [ADD INFOGRAPHIC: top blue Model: state → threshold → spike rectangle; two downward arrows to grey CPU simulation and blue Specialised hardware rectangles. Green Evidence label only under CPU: executed locally. Hardware label: not accessed.]

**Bottom banner / takeaway:** Distinguish the model, the implementation and the evidence.

**Notebook connection:** Notebook validation and limits.

**Teacher notes — 3 minutes**

- Use source device slides for imagery only if original attribution is retained.
- Avoid old specification tables and current-best-chip claims.
- Ask whether C code output proves low chip power; expected: no.
- Explain that memristor detail belongs in Backup for this lesson.
- No special device access, hardware learning or speedup was demonstrated.

### Slide 22 — Learning is a separate step

**Slide number and source:** New 22; original 46 + 50.  
**Action:** MERGE.  
**Title:** Learning is a separate step  
**Subtitle:** A spiking model can run with fixed parameters.

**Student-visible content**

- Our input, threshold and retention are set by the programmer.
- The code does not optimise weights or recognise a dataset.
- Some systems support learning; methods and support differ.
- Useful applications also need accuracy and robustness checks.

**Infographic / visual:** Two rectangles: Configure parameters → Run model. Optional Learning box in grey beneath, labelled Not implemented here.

**Bottom banner / takeaway:** Producing spikes is not the same as learning.

**Notebook connection:** Section 1 fixed-input model; Section 4 explanations.

**Teacher notes — 2 minutes**

- Locate the fixed input in C and the supplied command arguments.
- Ask where the loss function or training data appears; expected: nowhere.
- Do not present STDP as required for every spiking system.
- Changing a parameter by hand is an investigation, not a trained classifier.
- Preserve detailed learning algorithms in Backup only.

### Slide 23 — Where could this fit beside HPC?

**Slide number and source:** New 23; original 35.  
**Action:** HEAVY UPDATE.  
**Title:** Where could this fit beside HPC?  
**Subtitle:** Treat a specialist as one component of a complete workflow.

**Student-visible content**

- Use CPU/HPC resources to prepare data or simulate candidate models.
- Explore specialised event processing for a suitable workload.
- Validate output quality and response requirements.
- Include sensing, communication, setup and idle costs in comparisons.

**Infographic / visual:** [ADD INFOGRAPHIC: blue CPU preparation → grey Interface → blue Event-processing candidate → green Validated output. Add grey Compare with CPU baseline bracket. Dashed outline around candidate labelled conceptual, not executed.]

**Bottom banner / takeaway:** A useful component must improve a useful workflow.

**Notebook connection:** Sections 3–4 architectural interpretation.

**Teacher notes — 3 minutes**

- Replace broad climate/drug-discovery performance claims from the old slide.
- Explain hybrid integration conceptually without implying an installed device.
- Ask which stage the notebook executed: CPU simulation only.
- Require a baseline meeting the same output-quality and latency target.
- Bridge to later system topics: energy, reliability and software integration.

### Slide 24 — [DISCUSS] Would you investigate a specialist?

**Slide number and source:** New 24; original 51.  
**Action:** HEAVY UPDATE.  
**Title:** [DISCUSS] Would you investigate a specialist?  
**Subtitle:** Choose a baseline and identify the evidence still missing.

**Student-visible content**

- Case A: a sensor changes twice in a long interval.
- Case B: a signal changes on every sample.
- For each, predict message sparsity.
- Name a CPU/microcontroller baseline and one measurement needed.

**Infographic / visual:** Preserve original painting panel and aspect ratio; use the white panel for the two concise cases. Remove ML-versus-neuromorphic question.

**Bottom banner / takeaway:** A candidate architecture needs a workload and a fair comparison.

**Notebook connection:** Section 4 question 4.

**Teacher notes — 4 minutes**

- Allow two minutes in pairs, then two minutes for feedback.
- A offers sparse messages; B does not offer that particular advantage.
- Neither case alone proves the best processor choice.
- Accept latency, complete-system energy or detection quality as missing evidence.
- Do not make a safety-critical deployment claim from the toy sensor example.

### Slide 25 — Optional practice: explore a visual neuron

**Slide number and source:** New 25; original 58 + 59 + 60.  
**Action:** MERGE.  
**Title:** Optional practice: explore a visual neuron  
**Subtitle:** Use a guided external example after understanding the C model.

**Student-visible content**

- Open snnTorch Tutorial 2 from notebook Section 5.
- Use its editable Colab link and follow its setup there.
- Change one input-strength setting and keep a before/after figure.
- Submit the prediction, observation and two-sentence explanation.

**Infographic / visual:** One large text link with short task sequence; include EPFL exercises as a small alternative. Keep URLs in notes and clickable text, not tiny QR-only access.

**Bottom banner / takeaway:** External practice extends the lesson; it is optional.

**Notebook connection:** Section 5 optional external practice.

**Teacher notes — 2 minutes**

- Main link: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html .
- Alternative: https://neuronaldynamics.epfl.ch/exercises.html ; follow its exercise-site link.
- Use external setup guidance in a personal/Colab runtime; do not install packages on SciTech for this lesson.
- Different parameterisations and reset conventions can yield different spike counts.
- If access is unavailable, annotate supplied figures; do not invent executed results.

### Slide 26 — [TAKEAWAY] 4 essentials from Session 26

**Slide number and source:** New 26; original 54 + 55.  
**Action:** MERGE.  
**Title:** [TAKEAWAY] 4 essentials from Session 26  
**Subtitle:** Explain the mechanism and respect the limits of the evidence.

**Student-visible content**

- State: previous input can influence the next step.
- Spikes: threshold crossing produces a discrete output.
- Events: sparse changes can reduce messages.
- Evidence: CPU simulations do not prove chip speed or energy.

**Infographic / visual:** Four large blue/grey conceptual blocks, arranged 2×2. Preserve useful recap painting artwork at restrained size. Muted-green full-width banner below.

**Bottom banner / takeaway:** Match the workload, inspect the code, then measure the complete system.

**Notebook connection:** Sections 1–4 consolidation.

**Teacher notes — 2 minutes**

- Tie state to retention 0.8 versus 0.0.
- Tie spikes to the four-versus-two threshold result.
- Tie events to three versus eight messages.
- Ask what hardware efficiency was measured; expected: none.
- Keep four essentials only; avoid adding a dense reference table.

### Slide 27 — Exit question: what would you measure next?

**Slide number and source:** New 27; original 61.  
**Action:** UPDATE.  
**Title:** Exit question: what would you measure next?  
**Subtitle:** Turn a model result into a sensible evaluation question.

**Student-visible content**

- Name one parameter you changed and its observed effect.
- Name one measurement required before choosing real hardware.
- Bridge: energy, reliability and integration in larger HPC systems.

**Infographic / visual:** Preserve end-slide identity. One exit prompt and short next-session bridge; no new complex diagram.

**Bottom banner / takeaway:** A convincing result explains both what ran and what remains unknown.

**Notebook connection:** Section 4 written answers; Session 27 bridge.

**Teacher notes — 1 minutes**

- Collect one concise written response per student.
- Expected examples: threshold rises and spike count falls, or retention zero removes accumulated evidence.
- Accept whole-system energy, latency or matched detection quality as a next measurement.
- The official next session addresses exascale/post-exascale systems; do not silently reassign it to quantum.
- Close at the planned time; remaining external practice is optional.

## F. Ranked infographic priorities

| Rank | Visual | Slides | Why it matters |
|---|---|---|---|
| 1 | State → threshold → spike → reset loop | 6 | Makes the model concrete before code. |
| 2 | Aligned spike rasters for thresholds 1.0 and 1.5 | 9 | Links one parameter to an actual observed change. |
| 3 | Sensor samples, sparse messages and receiver state | 14–17 | Separates reading inputs from communicating changes. |
| 4 | Retention plus new input | 11 | Prevents confusing retention with leak or removal of input. |
| 5 | Model mapped to CPU and specialised hardware | 21 | Makes the simulation/hardware distinction explicit. |
| 6 | CPU preparation → interface → candidate → validation | 23 | Connects the neuromorphic concept to a complete HPC workflow. |

These are diagram specifications for future implementation; no infographics or PowerPoint slides have been generated in this update.

## G. PowerPoint implementation checklist

1. Preserve the original 108-slide file. Work on `HPC_M6S1_Neuromorphic_2026-27.pptx` only when editing is requested. Keep an original-number crosswalk.
2. Move the exact Backup groups above first, retaining original 76 as the divider. Quarantine or remove the four delete candidates after preserving the source.
3. Reuse cover 1 and agenda 2; correct 6.2 and M6.S2 labels in every surviving slide to M6S1 / Session 26.
4. Assemble the merged concept slides from the listed originals. Keep only one live copy of each concept after merging.
5. Replace the old code and ANN/SNN questions. Use only the tested `neuron.c` and `events.c` model snippets from the focused notebook.
6. Add the prediction, code, demo and one-parameter activity slides in the final order. Do not reinsert quantum, photonic or matrix-precision demonstrations from the earlier broad plan into this lesson.
7. Add five MCQs at new slides **3, 10, 13, 19 and 20**. Their correct answers are A, B, A, C and C respectively; keep these in notes only.
8. Preserve painting layouts on the opening question and discussion, and reuse the recap artwork where it improves legibility. Implement exact diagrams as editable vector shapes, not decorative illustrations.
9. Copy all five teacher-note bullets for each slide and keep answers hidden from the student-facing content. Add local timing to the notes.
10. Use notebook Sections 1–5 and current names `state`, `threshold`, `retention`, `before_reset`, `spike`, `after_reset`, `last` and `messages`. Do not refer to the superseded broad notebook’s E1–E5 labels.
11. Keep demonstrations honest: labelled local outputs are available, but SciTech is NOT RUN. Count messages rather than claiming processor speedup or energy savings. Restore baseline input and assertion after the alternating-input task.
12. Confirm the final plan has 27 slides, 80 planned minutes, five distributed MCQs, the four-essentials recap and an exit question. Verify any implemented deck visually before delivery; the PowerPoint remains unchanged now.
