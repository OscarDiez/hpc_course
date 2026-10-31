# M6S1 / Session 26 · Notebook-led slide restructuring plan

**IE University · BCSAI · High-Performance Computing · 2026–27**  
**Prepared 31 October 2026. PowerPoint remains unchanged.**

This plan uses the attached `HPC_S26_Quantum_Neuromorphic_Alternative_2026-27.pptx`. Despite its filename, its title says “6.2 Introduction to Neuromorphic computing” and it contains 108 slides of neuromorphic material. Original slide numbers below refer to that exact 108-slide file, before any moves. Preserve an untouched source copy so the numbers remain usable.

# Deliverable A · Completed notebook stage

- Pull request: https://github.com/OscarDiez/hpc_course/pull/54
- Notebook: https://github.com/OscarDiez/hpc_course/blob/teaching/m6s1-alternative-paradigms-2026/sessions/M6S1_quantum_neuromorphic_alternative/M6S1_quantum_neuromorphic_alternative.ipynb
- Raw URL for JupyterLab **File → Open from URL**: https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m6s1-alternative-paradigms-2026/sessions/M6S1_quantum_neuromorphic_alternative/M6S1_quantum_neuromorphic_alternative.ipynb
- Notebook commit completed before this plan: `80a5da20ff992326325e8fa677dc36b803126d6e`.
- Validation: https://github.com/OscarDiez/hpc_course/blob/teaching/m6s1-alternative-paradigms-2026/sessions/M6S1_quantum_neuromorphic_alternative/VALIDATION.md

## Repository conventions and syllabus mapping

Inspected root README, SciTech cluster guide and validation notes, M2S2, M3S1, M3S2 and M5S3 notebooks/README files, and the Session 11 C/Slurm demos. The new work follows `sessions/MxSy_topic/`, standalone `session_demos/`, a Python 3 orchestration kernel, inspectable C examples, prediction/check/explanation cycles, one-parameter changes and explicit execution status. The raw notebook embeds the exact source files and needs no checkout, download, specialized hardware or package installation. Optional plots require an existing Matplotlib installation.

| Official Session 26 objective | Notebook section and executable example | Classroom evidence |
|---|---|---|
| Basic neuromorphic ideas | E1 · Sparse sensor updates, `events()` | Equivalent CPU algorithms, checked frame by frame; operation counts |
| Neural state and events | E2 · A neuron remembers recent input, `lif()` | Discrete C state trace, threshold change, independent recurrence check |
| In-memory and photonic ideas | E3 · One matrix operation, several hardware mappings, `matvec()` | C matrix reference and quantization model, no device simulation |
| Quantum fundamentals | E4 · Quantum interference before measurement, `quantum()` | One ideal qubit on a CPU, phase-dependent probabilities, simulated samples |
| Limitations and hybrid HPC | E5 · The complete hybrid workflow | Explicit assumed cost model, break-even overhead and architecture decision |

**Validation:** all 17 notebook code cells executed locally in order using a Python harness. C compiled under GCC 13.3.0 with C11, `-Wall -Wextra -Werror`. All E1–E5 checks passed. Separate local tests covered parameter extremes, optional two-thread OpenMP agreement, six invalid C invocations, missing compiler, attempted compilation failure and mocked scheduler memory cleanup/reuse/missing-output handling. The notebook includes saved, labelled LOCAL_CPU outputs. JSON/cell structure and syntax were checked; a live Jupyter kernel/frontend and the URL-import UI were not exercised. The default outputs and the validation JSON record actual local results, not hypothetical measurements.

**SciTech: NOT RUN.** The notebook provides a separate CPU Slurm job: one node, one task, one CPU, 1 GB and a two-minute execution limit. The collection cell requires matching source/settings/job identity. It prints NOT RUN while outputs are unavailable. The batch route accommodates the repository’s recorded inherited-memory conflicts and module availability issues. MPI, GPU and specialized hardware execution are SKIPPED by design. Optional OpenMP checks numerical agreement only and needs two CPUs. Final SciTech acceptance requires an actual recorded job and results.

# Deliverable B · Session assessment

**Central goal:** students should explain why a particular workload might suit an alternative architecture, inspect a small executable example and identify the evidence needed to justify an architectural choice. They should distinguish an implementation mechanism from claims about speed, energy, accuracy and maturity.

**Syllabus priority:** Session 26 explicitly covers quantum, neuromorphic, photonic and in-memory computing as complements for selected HPC workloads. The existing deck provides reusable neuromorphic concepts but no adequate quantum or photonic teaching block. The main reading is *High Performance Computing: Applications and Frontiers*, Chapters 17–18. Supporting reading is the emerging-architecture material in *Foundations of High-Performance Computing*, Chapter 18. The official syllabus governs the scope, ahead of the old deck.

**Course continuity:** M3 supplies locality, measurement boundaries and application speedup. M4 supplies neural-network/inference context. M5 supplies heterogeneous workflows, environment identity and reproducibility. Session 27 covers exascale/post-exascale energy, reliability, memory bandwidth and scalability. Keep its detailed system discussion for that session.

**Existing deck issues:** repeated definitions, crowded comparisons, long unverified code, unsupported brain/device efficiency comparisons, old hardware tables, and a false distinction between ML and neuromorphic computing. SNNs are neural-network models; non-spiking networks can also handle time. Learning support varies by system. Memristors are optional implementation devices. The short C simulations replace the problematic Nengo examples. Reuse the IE identity and useful painting layouts without reusing obsolete claims.

**Timing:** 29 teaching slides, exactly 80 planned minutes including five distributed MCQs, predictions, demonstrations and the exit question. Prepare or submit the small job before teaching. If it is queued, show the saved LOCAL_CPU record with its label and return to actual SciTech results when available. Parameter investigations can finish after class. Do not use class time to explain orchestration syntax.

## Move to Backup first

Freeze numbering before editing. Create one Backup section after the final teaching slide. The groups below are disjoint and, with the reused slides and delete candidates, account for all 108 originals. “Backup” means preserve for reference, not endorse every original claim. Mark questionable content **REFERENCE ONLY · REQUIRES CORRECTION** until reviewed.

| Group | Exact original slide numbers | Why / required handling |
|---|---|---|
| Duplicates and overly detailed explanations | **8, 16, 38, 40, 41, 42, 43, 52, 53, 77, 78, 81, 89, 90, 101, 102** | Replaced by compact state/event and mechanism comparisons. Correct misleading static-ANN, automatically-learning and ML-versus-neuromorphic claims before optional reuse. |
| Advanced programming/reference material | **29, 30, 34, 47, 48, 82, 83, 85, 86, 87, 93, 94, 95, 96, 97, 98, 103, 104, 106, 107, 108** | Framework stacks, neuron taxonomies, training algorithms, device families and implementation detail exceed this introductory session. Keep as optional reference, with qualified capabilities. |
| Material belonging to other sessions | **88** | DNN acceleration on conventional platforms fits the earlier GPU/AI teaching. Do not repeat it here. |
| Optional case studies and history | **9, 12, 17, 24, 26, 27, 28, 44, 45, 79, 84, 91** | History, video, hardware tables and applications distract from the five verified examples. Treat brain comparisons and old chip figures as unverified. Original 26 needs reconstruction from primary sources before reuse. |
| Outdated questions and assessments | **4, 19, 20, 32, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75** | Old answer slides and quizzes repeat oversimplifications or old 6.2 numbering. Replace the live questions with the five specified MCQs. Revalidate any future question bank independently. |
| Other supplementary content | **5, 10, 39, 56, 58, 59, 60, 76, 80, 99** | Spare dividers, resource lists, old facts, AGI speculation and philosophical content. Original 56 needs primary-source checks and qualified measurement boundaries. Original 76 can serve as the Backup divider. Reuse layouts, not outdated text. |

**DELETE candidates, only after preserving the untouched source:** **33, 49, 57, 92**. Original 33 mixes framework names and unsupported hardware claims. Original 49 is incomplete code with a random “classifier” and a mislabeled dataset. Original 57 contains misleading deployment/learning advice. Original 92 presents a dated hardware list as current. Their teaching purpose is covered elsewhere. No deletion or movement has been performed.

**Reused original content/layout sources:** **1, 2, 3, 6, 7, 11, 13, 14, 15, 18, 21, 22, 23, 25, 31, 35, 36, 37, 46, 50, 51, 54, 55, 61, 100, 105**. These feed merged or heavily rewritten slides. Do not keep surplus source slides as additional live duplicates after merging. Preserve the original deck externally.

## Final order

Source numbers are original numbers. “NEW, layout from…” refers to a new content slide using an existing visual layout. It does not keep that old content in the teaching sequence. MCQ answers appear only in teacher notes.

| New # | Current source | Final title | Action | Notebook experiment | Infographic | Minutes |
|---|---|---|---|---|---|---:|
| 1 | 1 | M6S1 · Alternative computing paradigms | UPDATE | E1–E5 | Existing identity / native table, chart or text | 1 |
| 2 | 2 | What will we investigate? | HEAVY UPDATE | E1–E5 | Existing identity / native table, chart or text | 2 |
| 3 | 3 | [PREDICT] Fewer updates, less energy? | HEAVY UPDATE | E1 | Existing identity / native table, chart or text | 2 |
| 4 | 6 + 7 + 11 | Why alternative architectures? | MERGE | E1–E5 | NEW diagram | 3 |
| 5 | 18 | Four architectural ideas | HEAVY UPDATE | E1–E5 | NEW diagram | 3 |
| 6 | 13 + 14 + 15 + 21 | Neuromorphic computing | MERGE | E2 | NEW diagram | 3 |
| 7 | NEW | [PREDICT] How much work changes? | ADD | E1 | NEW diagram | 2 |
| 8 | NEW | [DEMO] Sparse updates, matching answers | ADD | E1 | Existing identity / native table, chart or text | 4 |
| 9 | 22 + 105 | A neuron with state | MERGE | E2 | NEW diagram | 3 |
| 10 | NEW | [PREDICT] When will the neuron fire? | ADD | E2 | Existing identity / native table, chart or text | 2 |
| 11 | NEW | [DEMO] Threshold changes the spike train | ADD | E2 | Existing identity / native table, chart or text | 4 |
| 12 | 31 | [MCQ] A zero state after firing | HEAVY UPDATE | E2 | Existing identity / native table, chart or text | 2 |
| 13 | 23 + 25 | From a model to a device | MERGE | E1–E5 | Existing identity / native table, chart or text | 2 |
| 14 | 36 + 37 + 100 | In-memory computing | MERGE | E3 | NEW diagram | 3 |
| 15 | NEW | [PREDICT] What counts as a fast result? | ADD | E3 | Existing identity / native table, chart or text | 2 |
| 16 | NEW, using the existing white master | Photonic computing | ADD | E3 | NEW diagram | 3 |
| 17 | NEW | [PREDICT] How much precision is enough? | ADD | E3 | Existing identity / native table, chart or text | 2 |
| 18 | NEW | [DEMO] Precision changes the answer | ADD | E3 | Existing identity / native table, chart or text | 4 |
| 19 | NEW, optional layout from original 10 | Quantum computing | ADD | E4 | NEW diagram | 4 |
| 20 | NEW | [PREDICT] Can phase change the result? | ADD | E4 | NEW diagram | 2 |
| 21 | NEW | [DEMO] Interference changes probabilities | ADD | E4 | Existing identity / native table, chart or text | 5 |
| 22 | NEW | [MCQ] Same probabilities, different states | ADD | E4 | Existing identity / native table, chart or text | 2 |
| 23 | NEW | Quantum workloads and practical limits | ADD | E4 | Existing identity / native table, chart or text | 3 |
| 24 | 35 + 46 + 50 | A specialist inside an HPC workflow | MERGE | E5 | NEW diagram | 3 |
| 25 | NEW | [DEMO] A fast kernel can lose overall | ADD | E5 | Existing identity / native table, chart or text | 4 |
| 26 | NEW | [MCQ] Should this kernel be offloaded? | ADD | E5 | Existing identity / native table, chart or text | 2 |
| 27 | 51 | [DISCUSS] Which architecture fits? | HEAVY UPDATE | E1–E5 | Existing identity / native table, chart or text | 4 |
| 28 | 54 + 55 | [TAKEAWAY] 4 essentials from Session 26 | MERGE | E1–E5 | Existing identity / native table, chart or text | 2 |
| 29 | 61 | Exit question · Beyond one accelerator | UPDATE | E1–E5 | Existing identity / native table, chart or text | 2 |

## Visual identity and implementation rules

Retain the 16:9 format, IE logos, painting artwork and master placements. The source uses Tahoma 28 pt for many titles; retain that title identity. Use a consistent sans-serif body around 22–24 pt when feasible. Rebuild tiny tables rather than shrinking them. Keep code to 2–4 meaningful lines. Use white backgrounds for technical diagrams, blue `#517C9C`, grey `#777777` and muted green `#668873`. Avoid decorative gradients, dense icon grids and complex circuitry. Technical diagrams should be editable vector elements when implemented. Original paintings remain raster artwork with preserved proportions. MCQ slides carry no visible answer cues.

## Complete individual slide specifications

### Slide 1 · M6S1 · Alternative computing paradigms

**Slide number and source:** New 1; original 1.  
**Action:** UPDATE.  
**Title:** M6S1 · Alternative computing paradigms  
**Subtitle:** Quantum, neuromorphic, photonic and in-memory computing in HPC.

**Student-visible content**

- Bachelor in Computer Science and Artificial Intelligence
- IE University · 2026–27 · Oscar Diez

**Infographic / visual:** Preserve the original cover artwork, IE logo positions and proportions. Replace the old 6.2 label with M6S1 and add Syllabus Session 26. Keep the title short and place the four paradigms in the subtitle.

**Bottom banner / takeaway:** Which architecture fits the workload?

**Notebook connection:** Notebook introduction and learning objectives.

**Teacher notes (1 minutes)**

- Explain that the session examines selected computing mechanisms and workload fit.
- Ask students to recall one reason a GPU was unsuitable for a small job.
- Expected answer: setup or transfer cost can exceed the kernel saving.
- Common misconception: every new processor improves every workload.
- Show the notebook tab and its execution context. Time: 1 minute.

### Slide 2 · What will we investigate?

**Slide number and source:** New 2; original 2.  
**Action:** HEAVY UPDATE.  
**Title:** What will we investigate?  
**Subtitle:** Five small experiments connect unfamiliar architectures to familiar computations.

**Student-visible content**

- Sparse sensor updates and a neuron with state
- A weighted sum with limited precision
- Quantum phase, interference and measurement
- The complete cost of offloading one kernel

**Infographic / visual:** A simple ordered list with a small notebook symbol. Remove the unrelated IBM rack caption and quotation. Preserve the existing white slide master.

**Bottom banner / takeaway:** Each architectural claim needs evidence.

**Notebook connection:** Introduction; E1–E5.

**Teacher notes (2 minutes)**

- Explain the prediction, execution, explanation and parameter-change routine.
- Ask which experiment will use a physical quantum computer.
- Expected answer: none. E4 is an ideal CPU simulation.
- Common misconception: a plotted quantum probability means a QPU ran.
- Point to LOCAL_CPU and the model labels. Time: 2 minutes.

### Slide 3 · [PREDICT] Fewer updates, less energy?

**Slide number and source:** New 3; original 3.  
**Action:** HEAVY UPDATE.  
**Title:** [PREDICT] Fewer updates, less energy?  
**Subtitle:** A sensor monitor receives only 20 changes during 1,000 frames.

**Student-visible content**

- Two CPU algorithms agree: 16,000 versus 20 weighted updates. What follows?
- A. A neuromorphic chip uses exactly 800 times less electricity.
- B. Updates decrease, but total cost still needs measurement.
- C. A CPU cannot execute the event algorithm.
- D. Event generation and stored state have zero cost.

**Infographic / visual:** Keep the original question-slide identity. Display the scenario above four large A–D choices. No answer highlighting.

**Bottom banner / takeaway:** OMIT on this question slide.

**Notebook connection:** E1, Checkpoint Q1. Use before the event-processing explanation.

**Teacher notes (2 minutes)**

- Correct answer: B. Counted useful operations do not measure total energy.
- Ask students to vote before introducing neuromorphic hardware.
- A is wrong because energy also depends on memory, communication, idle power and implementation.
- C is wrong because this notebook runs the event algorithm on a CPU. D omits state and event-generation costs.
- Collect votes without revealing the answer until E1 discussion. Time: 2 minutes.

### Slide 4 · Why alternative architectures?

**Slide number and source:** New 4; original 6 + 7 + 11.  
**Action:** MERGE.  
**Title:** Why alternative architectures?  
**Subtitle:** Data movement and workload structure shape useful performance.

**Student-visible content**

- Parallelism helps only the work that can use it.
- Moving data can cost more than the arithmetic.
- Specialists change how selected operations execute.
- The useful comparison covers the full application.

**Infographic / visual:** [ADD INFOGRAPHIC: a blue Compute rectangle and grey Memory rectangle linked by two thick labelled Data arrows. Beneath, a muted-green Local state + compute rectangle with short internal arrows. Caption: architectural idea, not an energy measurement.]

**Bottom banner / takeaway:** Data movement belongs in the performance model.

**Notebook connection:** E1 operation counts; E3 stored weights; E5 complete cost.

**Teacher notes (3 minutes)**

- Connect memory locality to M3 and heterogeneous execution to M5.
- Ask whether Amdahl’s law disappears when the kernel uses a specialist.
- Expected answer: no, unchanged phases still limit application benefit.
- Correct the old claim that conventional CMOS necessarily loses all data when power is off. Volatility depends on the memory technology.
- Use the diagram instead of the old historical scaling charts. Time: 3 minutes.

### Slide 5 · Four architectural ideas

**Slide number and source:** New 5; original 18.  
**Action:** HEAVY UPDATE.  
**Title:** Four architectural ideas  
**Subtitle:** These categories describe different mechanisms and can overlap.

**Student-visible content**

- Neuromorphic: neural state and event communication
- In-memory: computation where weights or data reside
- Photonic: selected computation using optical signals
- Quantum: amplitudes, interference and measurement

**Infographic / visual:** [ADD INFOGRAPHIC: four-row comparison table. Columns: Idea, Main operation, Notebook evidence. Rows connect neuromorphic to E1/E2, in-memory to E3, photonic to E3, quantum to E4. White background, blue labels, grey mechanisms, green evidence. Footnote: photonic in-memory and photonic neuromorphic systems can overlap.]

**Bottom banner / takeaway:** Architecture names describe mechanisms, not universal winners.

**Notebook connection:** Learning objectives; E1–E4 evidence boundaries.

**Teacher notes (3 minutes)**

- Explain each row in one sentence. An implementation can belong to more than one row.
- Ask whether a photonic matrix engine must implement a spiking network.
- Expected answer: no. Weighted arithmetic and spiking dynamics are different concepts.
- Avoid the old comparison’s sequential CPU, binary-versus-spike and automatic-learning dichotomies.
- Point forward to the same y=W x example for two hardware mappings. Time: 3 minutes.

### Slide 6 · Neuromorphic computing

**Slide number and source:** New 6; original 13 + 14 + 15 + 21.  
**Action:** MERGE.  
**Title:** Neuromorphic computing  
**Subtitle:** Neural systems inspire local state, weighted connections and communication.

**Student-visible content**

- Neurons hold a changing internal state.
- Connections determine how inputs influence that state.
- Spikes can communicate events between neurons.
- Learning requires a separate supported rule.

**Infographic / visual:** [ADD INFOGRAPHIC: three blue Input event rectangles connect through grey Weight rectangles to a blue Neuron state rectangle. A green Spike output rectangle sits to the right. A small grey local-state loop returns to the neuron. Use arrows labelled weighted input and emitted event.]

**Bottom banner / takeaway:** A neuron can remember input without learning new weights.

**Notebook connection:** E2, A neuron remembers recent input; C function lif().

**Teacher notes (3 minutes)**

- Explain stored state versus parameter changes, using an alarm accumulating evidence.
- Ask what remains after an input disappears.
- Expected answer: the state may persist and decay, depending on the model.
- Do not imply every neuromorphic design uses spikes, memristors, exact brain models or on-chip learning.
- Keep detailed biological vocabulary in Backup. Time: 3 minutes.

### Slide 7 · [PREDICT] How much work changes?

**Slide number and source:** New 7; original NEW.  
**Action:** ADD.  
**Title:** [PREDICT] How much work changes?  
**Subtitle:** Only one of 16 sensor channels changes at an event.

**Student-visible content**

- 1,000 frames and 16 channels
- Dense path recalculates all 16 weighted inputs per frame.
- Event path updates its stored weighted sum once per change.
- Predict counts when an event arrives every 50 frames, then every 100.

**Infographic / visual:** [ADD INFOGRAPHIC: a short grey frame timeline with isolated blue event marks, above a dense grid of 16 grey channel rows. Highlight one green changed cell at each event. Label dense recalculation and supplied changes.]

**Bottom banner / takeaway:** Predict the useful work before counting it.

**Notebook connection:** E1 prediction and C function events().

**Teacher notes (2 minutes)**

- Explain that the event path receives changes already identified by the input source.
- Ask for both update counts, without claiming a timing ratio.
- Expected: dense 16,000, event 20 at period 50, event 10 at period 100.
- Common misconception: acquiring or constructing the event stream is free.
- Show only the prediction, keeping the result cell hidden. Time: 2 minutes.

### Slide 8 · [DEMO] Sparse updates, matching answers

**Slide number and source:** New 8; original NEW.  
**Action:** ADD.  
**Title:** [DEMO] Sparse updates, matching answers  
**Subtitle:** Two CPU implementations agree at every simulated frame.

**Student-visible content**

- Run E1 and inspect PASS checks.
- Period 50: 16,000 dense updates and 20 event updates.
- Period 100: 16,000 dense updates and 10 event updates.
- The counts describe weighted updates, not measured electricity.

**Infographic / visual:** Use a native two-row table with period, dense updates and event updates. Add a green Equality checked label. Caption: author-tested LOCAL_CPU simulation; replace context only after a real class run.

**Bottom banner / takeaway:** Sparsity reduces work when the algorithm can exploit it.

**Notebook connection:** E1 result table; raw events records; optional Section 6 plot.

**Teacher notes (4 minutes)**

- Run the existing E1 cell or show its saved, visibly labelled local record if the cluster is queued.
- Ask how we know the optimization preserved the result.
- Expected: equality at each frame and an independently checked accumulated checksum.
- Different periods generate different input streams, so compare algorithms within a run rather than cross-period checksums.
- Reveal Q1 answer B and invite the period change. Time: 4 minutes.

### Slide 9 · A neuron with state

**Slide number and source:** New 9; original 22 + 105.  
**Action:** MERGE.  
**Title:** A neuron with state  
**Subtitle:** A simple recurrence combines new input with a decaying memory.

**Student-visible content**

- Before reset: u = 0.8v + 0.3
- Emit a spike when u reaches the threshold.
- After a spike, this model resets v to zero.
- Otherwise, the next state is u.

**Infographic / visual:** [ADD INFOGRAPHIC: grey Previous state enters a blue Multiply by 0.8 box, then a blue Add 0.3 box and a Threshold decision. A green Spike arrow leads to Reset 0. A grey No spike arrow leads to Keep u. Both return to Stored v. Label discrete teaching model.]

**Bottom banner / takeaway:** The firing decision uses the state before reset.

**Notebook connection:** E2 recurrence; C function lif().

**Teacher notes (3 minutes)**

- Explain all three operations with one input step. This is a simplified discrete model.
- Ask whether the weight changes when a spike occurs.
- Expected answer: no, this example contains no learning rule.
- Common misconception: a reset erases evidence that a spike occurred. The output event remains.
- Use the recurrence diagram rather than neuron-model taxonomy. Time: 3 minutes.

### Slide 10 · [PREDICT] When will the neuron fire?

**Slide number and source:** New 10; original NEW.  
**Action:** ADD.  
**Title:** [PREDICT] When will the neuron fire?  
**Subtitle:** The threshold sets how much accumulated evidence triggers the output.

**Student-visible content**

- Start with v=0, decay=0.8 and input=0.3.
- The first states before reset are 0.3, 0.54 and 0.732.
- Predict what happens at threshold 1.0.
- Will threshold 1.3 give more or fewer spikes in twelve steps?

**Infographic / visual:** A sparse state table with step and pre-reset value. Leave later rows blank for prediction. Add a horizontal threshold guide only if it improves projection readability.

**Bottom banner / takeaway:** Compare thresholds while keeping the input fixed.

**Notebook connection:** E2 prediction; SETTINGS threshold.

**Teacher notes (2 minutes)**

- Ask students to reason about accumulation instead of doing lengthy arithmetic.
- Ask whether all positive thresholds must eventually produce a spike.
- Expected: no. This subthreshold recurrence approaches 1.5 with constant input.
- Common misconception: nonzero input always guarantees firing at any threshold.
- Let students predict direction before opening the trace. Time: 2 minutes.

### Slide 11 · [DEMO] Threshold changes the spike train

**Slide number and source:** New 11; original NEW.  
**Action:** ADD.  
**Title:** [DEMO] Threshold changes the spike train  
**Subtitle:** The same twelve input steps produce different output events.

**Student-visible content**

- Threshold 1.0 gives two spikes.
- Threshold 1.3 gives one spike.
- Read before_reset, spike and after_reset separately.
- Try threshold 1.6 and explain the result.

**Infographic / visual:** Plot the actual pre-reset trace with a blue state line and grey threshold line. Use green vertical ticks for spikes. If showing threshold 1.3, place it below as a second aligned trace. Caption: discrete CPU simulation, twelve steps.

**Bottom banner / takeaway:** State and threshold determine timing.

**Notebook connection:** E2 table and TRACE output; parameter threshold.

**Teacher notes (4 minutes)**

- Run E2 and locate the first threshold crossing. Local default first crossing is step 4, using zero-based indexing.
- Ask why the post-reset state can be zero on a firing step.
- Expected: the model emits the event and then resets. Threshold 1.6 produces no spike because the limiting subthreshold value is 1.5.
- Do not generalize this specific firing-count result to arbitrary noisy input streams or biological neurons.
- Demonstrate one threshold change if results are available. Time: 4 minutes.

### Slide 12 · [MCQ] A zero state after firing

**Slide number and source:** New 12; original 31.  
**Action:** HEAVY UPDATE.  
**Title:** [MCQ] A zero state after firing  
**Subtitle:** The trace shows a threshold crossing followed by reset.

**Student-visible content**

- Before reset: 1.008, threshold: 1.0, after reset: 0. What should the record show?
- A. No spike, because the post-reset state is zero.
- B. A learned weight change.
- C. One spike and a post-reset state of zero.
- D. A fault, because a neuron cannot store state.

**Infographic / visual:** Use the existing A–D question layout. Keep the trace in a single highlighted line above the options. No answer or coloured correct option on the student slide.

**Bottom banner / takeaway:** OMIT on this question slide.

**Notebook connection:** E2 Checkpoint Q2.

**Teacher notes (2 minutes)**

- Correct answer: C. The threshold check precedes reset. The displayed 1.008 is a rounded illustrative trace value.
- Ask students to identify which column records the event.
- A confuses the stored state after the step with the output event.
- B invents learning and D incorrectly denies state.
- Use feedback only after votes. Time: 2 minutes.

### Slide 13 · From a model to a device

**Slide number and source:** New 13; original 23 + 25.  
**Action:** MERGE.  
**Title:** From a model to a device  
**Subtitle:** Software simulations and specialist processors provide different evidence.

**Student-visible content**

- A CPU simulation helps inspect the model.
- A neuromorphic processor implements selected neural mechanisms.
- Intel Loihi 2 is a digital neuromorphic research example.
- Access, learning support and workload results need separate checks.

**Infographic / visual:** Preserve one useful chip image with its source credit. Place CPU simulation and specialist hardware in two large labelled columns. Avoid specification tables and neuron-count comparisons with brains.

**Bottom banner / takeaway:** A simulated spike train does not measure hardware efficiency.

**Notebook connection:** E1/E2 evidence labels; Section 7 optional-feature statuses.

**Teacher notes (2 minutes)**

- Explain the notebook’s lack of physical neuromorphic execution.
- Ask what extra evidence a claim of lower power would require.
- Expected: the same useful task and quality, a defined measurement boundary and actual energy or power data.
- Do not imply new memory materials or on-chip learning are mandatory. Cite Intel’s official neuromorphic page in eventual notes.
- Retain chip detail as optional reference. Time: 2 minutes.

### Slide 14 · In-memory computing

**Slide number and source:** New 14; original 36 + 37 + 100.  
**Action:** MERGE.  
**Title:** In-memory computing  
**Subtitle:** Stored weights can participate directly in a weighted calculation.

**Student-visible content**

- The example operation is y=W x.
- A resistive implementation can encode weights as conductances.
- Applied voltages create weighted currents that sum at outputs.
- Programming, input conversion and readout still matter.

**Infographic / visual:** [ADD INFOGRAPHIC: a large 4×4 rectangular crossbar. Blue input arrows x1…x4 enter rows. Grey weight/conductance cells mark intersections. Muted-green output arrows y1…y4 leave columns. Put input encoding and output readout boxes outside the array. Caption: resistive in-memory example, idealized mechanism.]

**Bottom banner / takeaway:** Keeping weights near computation can reduce movement.

**Notebook connection:** E3 hardware concept and matvec(). Orient row/column labels consistently with y_i=sum_j W_ij x_j.

**Teacher notes (3 minutes)**

- Explain current contributions with one output dot product, without device physics.
- Ask whether the crossbar has to generate spikes.
- Expected answer: no, this example computes weighted sums.
- Correct the old suggestion that a memristor automatically learns a spoken word. Learning needs a system and update rule.
- Show only the array-level mechanism; move device structures to Backup. Time: 3 minutes.

### Slide 15 · [PREDICT] What counts as a fast result?

**Slide number and source:** New 15; original NEW.  
**Action:** ADD.  
**Title:** [PREDICT] What counts as a fast result?  
**Subtitle:** An optical matrix engine still needs input and output handling.

**Student-visible content**

- Which comparison supports choosing the engine?
- A. Optical propagation time alone.
- B. Number of weights alone.
- C. Peak operations per second without an accuracy requirement.
- D. End-to-end latency, energy and accuracy for the same workload.

**Infographic / visual:** Use a painting-style question layout derived from the original deck. Keep the four options on a plain white readable region. Do not add an answer cue.

**Bottom banner / takeaway:** OMIT on this question slide.

**Notebook connection:** E3 Checkpoint Q3, used before photonic mechanisms.

**Teacher notes (2 minutes)**

- Correct answer: D. Comparable useful output and the complete execution path define the comparison.
- Ask students to recall kernel versus total GPU timing from M2.
- A excludes encoding/readout and B does not describe performance.
- C ignores precision and supporting computation. These remain relevant even with a high peak rate.
- Take a quick vote, explain after the next two slides. Time: 2 minutes.

### Slide 16 · Photonic computing

**Slide number and source:** New 16; original NEW, using the existing white master.  
**Action:** ADD.  
**Title:** Photonic computing  
**Subtitle:** Optical signals can implement selected weighted transformations.

**Student-visible content**

- Encode inputs into an optical signal.
- A configured optical structure combines weighted contributions.
- Detect and read the outputs.
- Precision, calibration and conversion affect the useful result.

**Infographic / visual:** [ADD INFOGRAPHIC: left-to-right rectangular chain Electronic x, Input encoding, Optical weighted operation, Detection/readout, Electronic y. Use blue for the optical operation, grey for interfaces and green for the output. Place a small grey control/calibration box above the optical block with a downward arrow. Caption: one possible photonic matrix workflow.]

**Bottom banner / takeaway:** The optical operation is part of the full system.

**Notebook connection:** E3 hardware concept; the CPU arithmetic illustrates y=W x only.

**Teacher notes (3 minutes)**

- Explain that photonic computing here means selected classical optical computation. Photons can also implement quantum systems, but this slide describes a classical matrix operation.
- Ask whether the notebook measures optical propagation or converter energy.
- Expected answer: neither. It runs arithmetic on a CPU and explores an error model.
- Do not claim that light makes computation free or that all photonic engines have this exact layout. Cite primary photonic in-memory research in eventual notes.
- Reveal Q3 answer D and point to the conversion blocks. Time: 3 minutes.

### Slide 17 · [PREDICT] How much precision is enough?

**Slide number and source:** New 17; original NEW.  
**Action:** ADD.  
**Title:** [PREDICT] How much precision is enough?  
**Subtitle:** Rounding stored weights can change a weighted sum.

**Student-visible content**

- Use four inputs and four outputs.
- Reference weights: W[i,j]=(i+1)(j+1)/17.
- Quantized weights use 2^bits−1 equal intervals over [0,1].
- Predict the change from 3 bits to 8 bits.

**Infographic / visual:** Two aligned number lines from 0 to 1. Mark a coarse grey grid and a finer blue grid, with one green original weight and arrows to its rounded positions. Use three labels at most per line.

**Bottom banner / takeaway:** Faster arithmetic still needs adequate output quality.

**Notebook connection:** E3 prediction; SETTINGS bits.

**Teacher notes (2 minutes)**

- Explain that the model rounds weights only and keeps all other aspects fixed.
- Ask whether more bits guarantee monotonically smaller error in every individual rounded value.
- Expected: the bound tightens, but individual errors need not improve at every bit count.
- No conclusion about device noise, drift or classification accuracy follows from this small matrix alone.
- Show the short two-line matvec snippet instead of the full C source. Time: 2 minutes.

### Slide 18 · [DEMO] Precision changes the answer

**Slide number and source:** New 18; original NEW.  
**Action:** ADD.  
**Title:** [DEMO] Precision changes the answer  
**Subtitle:** The reference and quantized calculations have explicit numerical checks.

**Student-visible content**

- Run E3 and inspect reference, quantized and absolute error.
- At 3 bits, maximum error is about 0.02437.
- At 8 bits, these constructed weights are exact.
- The analytical error bound is 0.5/(2^bits−1).

**Infographic / visual:** A native table with bits, maximum error and bound. Reuse the notebook result values, with a small footnote: CPU arithmetic and quantization model. Show one output row in larger type, avoiding a full source listing.

**Bottom banner / takeaway:** Precision is part of the architectural choice.

**Notebook connection:** E3 result table, MATRIX records and independent reference checks.

**Teacher notes (4 minutes)**

- Run the E3 analysis cell and explain one row.
- Ask why the eight-bit result is exactly representable in this example.
- Expected: 255/17=15, so the chosen weights land on the quantization grid. This is specific to the example.
- Do not present zero error here as a general property of eight-bit hardware or inference.
- Invite a bit-count change and retain the error bound as the check. Time: 4 minutes.

### Slide 19 · Quantum computing

**Slide number and source:** New 19; original NEW, optional layout from original 10.  
**Action:** ADD.  
**Title:** Quantum computing  
**Subtitle:** Amplitudes combine before measurement produces an outcome.

**Student-visible content**

- A qubit has amplitudes associated with outcomes 0 and 1.
- Gates transform those amplitudes.
- Relative phase affects how amplitudes combine.
- Measurement produces one sampled outcome.

**Infographic / visual:** [ADD INFOGRAPHIC: a blue two-entry Amplitudes rectangle enters a Gate rectangle, then an Interference rectangle, then a green One outcome rectangle labelled 0 or 1. Beneath the amplitudes box show |a|²+|b|²=1. Avoid a Bloch sphere and the phrase all answers at once.]

**Bottom banner / takeaway:** Algorithms shape probabilities before measurement.

**Notebook connection:** E4 introduction and evidence boundary.

**Teacher notes (4 minutes)**

- Introduce amplitudes qualitatively before complex-number syntax.
- Ask what one measurement returns.
- Expected answer: one outcome for this measured qubit, not its two complex amplitudes.
- This single-qubit example demonstrates interference. It does not demonstrate entanglement, universal advantage or scalable quantum computing.
- Use the deck’s existing painting/divider style only as a framing element. Time: 4 minutes.

### Slide 20 · [PREDICT] Can phase change the result?

**Slide number and source:** New 20; original NEW.  
**Action:** ADD.  
**Title:** [PREDICT] Can phase change the result?  
**Subtitle:** The final gate combines amplitudes that carry a relative phase.

**Student-visible content**

- Begin at |0⟩.
- Apply H, a phase φ, then H.
- Compare φ=0 and φ=π.
- Before the last H, both cases have outcome probabilities 1/2 and 1/2.

**Infographic / visual:** [ADD INFOGRAPHIC: one horizontal circuit wire with labelled rectangles H, Phase φ, H, Measurement. Add two annotation brackets before the final H: equal basis probabilities and different relative phase. Use blue gates, grey wire and green measurement.]

**Bottom banner / takeaway:** Equal probabilities can hide different phases.

**Notebook connection:** E4 prediction; quantum() and SETTINGS phase.

**Teacher notes (2 minutes)**

- Ask students to vote same versus different final distributions.
- Explain that phase does not change intermediate basis probabilities in this circuit.
- Expected: the final H makes the phase difference observable in the outcome distribution.
- Do not say the phase gate itself changes those intermediate probabilities.
- Keep the result hidden until the demo. Time: 2 minutes.

### Slide 21 · [DEMO] Interference changes probabilities

**Slide number and source:** New 21; original NEW.  
**Action:** ADD.  
**Title:** [DEMO] Interference changes probabilities  
**Subtitle:** The CPU simulation checks its result against an analytical formula.

**Student-visible content**

- φ=0 gives P(1)=0.
- φ=π/2 gives P(1)=1/2.
- φ=π gives P(1)=1.
- Compare exact probabilities with 200 seeded simulated shots.

**Infographic / visual:** A native probability chart with three phase categories and blue P(1) bars. Beside it, show the actual seeded counts 0, 105 and 200 out of 200 from author LOCAL_CPU validation. Clearly label exact calculation and CPU-simulated samples separately. Update sample counts only from a fresh run.

**Bottom banner / takeaway:** Interference changes outcomes before sampling.

**Notebook connection:** E4 table and seeded shots; quantum_normalization and quantum_analytic_probability checks.

**Teacher notes (5 minutes)**

- Run E4. With seed 26 and the default row order, the local sample counts were 0, 105 and 200.
- Ask why 105 rather than exactly 100 is acceptable in the middle case.
- Expected: finite sampling fluctuates around the ideal probability. The analytic calculation remains 1/2.
- Explain that changing row order or sampling choices can change the seeded counts. Never label this QPU output.
- Point to the CPU provenance and allow one phase change. Time: 5 minutes.

### Slide 22 · [MCQ] Same probabilities, different states

**Slide number and source:** New 22; original NEW.  
**Action:** ADD.  
**Title:** [MCQ] Same probabilities, different states  
**Subtitle:** A final H produces different distributions after a phase operation.

**Student-visible content**

- Two states had equal basis probabilities before the last H. What explains the different final results?
- A. Relative phase affects interference at the final gate.
- B. Measurement reads every amplitude.
- C. The CPU simulation used a quantum processor.
- D. Equal intermediate probabilities always imply identical states.

**Infographic / visual:** Use a plain question layout with a small H–Phase–H circuit reminder. No answer highlighting.

**Bottom banner / takeaway:** OMIT on this question slide.

**Notebook connection:** E4 Checkpoint Q4.

**Teacher notes (2 minutes)**

- Correct answer: A. The circuit combines amplitudes whose relative phase differs.
- Ask students to name the gate that makes the difference visible in basis probabilities.
- Expected: the final H. B overstates what measurement returns.
- C mislabels the execution environment and D ignores phase information.
- Read feedback after the vote. Time: 2 minutes.

### Slide 23 · Quantum workloads and practical limits

**Slide number and source:** New 23; original NEW.  
**Action:** ADD.  
**Title:** Quantum workloads and practical limits  
**Subtitle:** A useful quantum subroutine needs an appropriate problem and full validation.

**Student-visible content**

- Quantum chemistry and materials motivate quantum simulation research.
- Factoring illustrates an algorithm with cryptographic significance.
- HPC can support preparation, simulation, control and analysis.
- Errors, circuit resources and repeated measurements constrain useful execution.

**Infographic / visual:** A small workload table: Chemistry/materials, Factoring, Hybrid variational research. Columns: Motivation and What must be checked. Use no speedup figures or commercial-readiness dates. Add a grey boundary note: E4 illustrates one qubit only.

**Bottom banner / takeaway:** A quantum circuit needs an algorithm and a resource budget.

**Notebook connection:** E4 memory estimate table; E5 measurement boundary discussion.

**Teacher notes (3 minutes)**

- Explain application motivation without implying present useful advantage in every example.
- Ask why the notebook estimates full state-vector memory instead of allocating it.
- Expected: the 16×2^n-byte growth rapidly exceeds small classroom resources.
- Some structured circuits admit other efficient classical simulations. The estimate does not prove quantum advantage. Factoring requires resources far beyond this one-qubit example.
- Connect to the syllabus quantum chapter rather than introducing detailed algorithms. Time: 3 minutes.

### Slide 24 · A specialist inside an HPC workflow

**Slide number and source:** New 24; original 35 + 46 + 50.  
**Action:** MERGE.  
**Title:** A specialist inside an HPC workflow  
**Subtitle:** Other application phases remain when one kernel moves to an accelerator.

**Student-visible content**

- Prepare the input on the classical system.
- Encode or transfer data and configure the specialist.
- Execute the selected subroutine.
- Read, validate and use its result.

**Infographic / visual:** [ADD INFOGRAPHIC: CPU preparation, Transfer/setup, Specialist kernel, Readout/validation rectangles connected in order. Blue CPU/kernel blocks, grey interfaces, muted-green validated result. A bracket underneath spans all blocks and reads End-to-end boundary. Add a return arrow from validation to preparation for iterative workflows.]

**Bottom banner / takeaway:** Compare the complete useful computation.

**Notebook connection:** E5 full workflow model, following E1–E4.

**Teacher notes (3 minutes)**

- Explain the shared hybrid pattern without claiming all devices use the same software interface.
- Ask which phases disappear when the specialist kernel becomes ten times faster.
- Expected: none automatically. Other work and transfer/setup remain.
- Replace the old unsupported implication that neuromorphic devices are already ideal for full climate or drug-discovery workloads.
- Connect to M5 workflow reproducibility and M3 application speedup. Time: 3 minutes.

### Slide 25 · [DEMO] A fast kernel can lose overall

**Slide number and source:** New 25; original NEW.  
**Action:** ADD.  
**Title:** [DEMO] A fast kernel can lose overall  
**Subtitle:** The model includes all phases inside the chosen boundary.

**Student-visible content**

- CPU baseline: 20 ms other work + 80 ms kernel.
- Assumed specialist kernel: 8 ms.
- With 5 ms overhead: total 33 ms.
- With 80 ms overhead: total 108 ms.

**Infographic / visual:** Native horizontal stacked bars with identical scale: CPU 20+80, hybrid 20+5+8, hybrid 20+80+8. Blue kernel, grey other work, muted-green overhead. Caption prominently: illustrative model, assumed milliseconds, not measured device timings.

**Bottom banner / takeaway:** Offloading helps only when its full cost fits.

**Notebook connection:** E5 result table and cost formula; SETTINGS overhead_ms.

**Teacher notes (4 minutes)**

- Ask for the total before revealing each bar.
- Run E5 and compare with 100 ms. The corresponding model speedups are about 3.03 and 0.926.
- Expected break-even overhead: 80−8=72 ms. Lower is faster under these assumptions.
- Queue waiting, hardware error mitigation and changed output quality are outside this simplified model and may need explicit terms in real comparisons.
- Emphasize that the C simulation runtimes did not supply these assumed timings. Time: 4 minutes.

### Slide 26 · [MCQ] Should this kernel be offloaded?

**Slide number and source:** New 26; original NEW.  
**Action:** ADD.  
**Title:** [MCQ] Should this kernel be offloaded?  
**Subtitle:** The application includes unchanged work and a costly interface.

**Student-visible content**

- Kernel: 80 to 8 ms. Other work: 20 ms. Transfer/setup: 80 ms. CPU baseline: 100 ms.
- A. The whole application is ten times faster.
- B. Total 108 ms, slower than the CPU baseline.
- C. Total 8 ms, because other phases disappear.
- D. Total 28 ms, because data transfer is free.

**Infographic / visual:** Use the same question layout as previous MCQs. A one-line timing scenario replaces a large diagram. No visible correct answer.

**Bottom banner / takeaway:** OMIT on this question slide.

**Notebook connection:** E5 Checkpoint Q5.

**Teacher notes (2 minutes)**

- Correct answer: B. 20+8+80=108 ms.
- Ask whether a larger kernel-only speedup guarantees a worthwhile application result.
- Expected: no, although a sufficiently large saving can offset a fixed overhead.
- A confuses kernel and application. C removes required work. D ignores transfer/setup.
- Read feedback and move to architecture choice. Time: 2 minutes.

### Slide 27 · [DISCUSS] Which architecture fits?

**Slide number and source:** New 27; original 51.  
**Action:** HEAVY UPDATE.  
**Title:** [DISCUSS] Which architecture fits?  
**Subtitle:** Choose a candidate and name the evidence that could change your decision.

**Student-visible content**

- A battery monitor receives sparse changes and needs prompt alarms.
- An inference service repeatedly evaluates a fixed matrix at a specified tolerance.
- A materials researcher proposes a quantum subroutine.
- For each, name a baseline and one missing measurement.

**Infographic / visual:** Preserve the painting-style left panel from original 51. Replace the false ML-versus-neuromorphic prompt with the three scenarios on the white right panel. Keep the artwork at its original aspect ratio.

**Bottom banner / takeaway:** Workload fit starts the comparison.

**Notebook connection:** E1/E2 sensing, E3 matrix precision, E4 quantum limits, E5 full cost; Section 7 handover.

**Teacher notes (4 minutes)**

- Allow one minute in pairs, then take two concise explanations.
- Expected sensing choice: compare an event-based CPU/microcontroller baseline with a neuromorphic candidate at the required detection quality and latency.
- Expected inference choice: compare CPU/GPU and possible in-memory/photonic implementations at matched precision, throughput and end-to-end energy.
- Expected materials choice: justify the quantum algorithm and resource/accuracy requirements against a strong classical baseline. These are candidates, not predetermined winners.
- Accept well-supported alternatives. Reserve one additional minute for the final response. Time: 4 minutes.

### Slide 28 · [TAKEAWAY] 4 essentials from Session 26

**Slide number and source:** New 28; original 54 + 55.  
**Action:** MERGE.  
**Title:** [TAKEAWAY] 4 essentials from Session 26  
**Subtitle:** Alternative architectures help when their mechanism fits the useful work.

**Student-visible content**

- Workload fit: identify the operation worth accelerating.
- Events and state: sparse input can reduce useful work.
- Accuracy and evidence: label simulations, counts and models.
- Hybrid cost: include encoding, transfer and readout.

**Infographic / visual:** Preserve a restrained part of the existing recap artwork. Use four large conceptual blocks arranged vertically or as a clean 2×2 recap with no dashboard styling. Each block has one short label and one sentence. Blue headings, grey text, muted-green bottom banner.

**Bottom banner / takeaway:** The best architecture delivers the required result within the full budget.

**Notebook connection:** Notebook Section 7 four essentials.

**Teacher notes (2 minutes)**

- Restate one demonstrated example for each block.
- Ask which result actually measured an alternative processor.
- Expected answer: none. We executed CPU concept examples and calculated explicit models.
- Do not convert operation-count savings, exact ideal probabilities or illustrative milliseconds into device-performance claims.
- Finish the recap before opening the exit question. Time: 2 minutes.

### Slide 29 · Exit question · Beyond one accelerator

**Slide number and source:** New 29; original 61.  
**Action:** UPDATE.  
**Title:** Exit question · Beyond one accelerator  
**Subtitle:** What must stay reliable and affordable when the complete system grows?

**Student-visible content**

- Choose one specialist from today’s session.
- Name one benefit and one system cost or reliability risk.
- Session 27: exascale and post-exascale systems.

**Infographic / visual:** Keep the original end-slide identity with a simple question. Add a small text bridge to energy, reliability, memory movement and software scalability. No new complex diagram.

**Bottom banner / takeaway:** A useful component must work inside a useful system.

**Notebook connection:** Notebook Section 7 bridge; next syllabus Session 27.

**Teacher notes (2 minutes)**

- Ask for a one-sentence exit answer rather than a new technical discussion.
- Expected examples include fast matrix arithmetic with conversion cost, event efficiency with input/quality constraints, or a quantum candidate with error/resource limits.
- Use two answers to introduce next session’s energy and reliability themes.
- Common misconception: a component-level result guarantees whole-system scalability.
- Use the extra minute for a final question or collecting answers. Time: 2 minutes.

## Ranked infographic priorities

| Rank | Visual | Slides | Purpose |
|---|---|---|---|
| 1 | Full hybrid workflow and matched stacked cost bars | 24–25 | Connect component speed to useful application outcome and explicit measurement boundaries. |
| 2 | Neuron state, threshold and reset with actual trace | 9–11 | Make temporal state and spike emission concrete without complex syntax. |
| 3 | H–Phase–H circuit with exact probabilities and simulated shots | 20–21 | Explain phase/interference and distinguish probability from a measurement sample. |
| 4 | Matrix weights with resistive and optical mappings | 14, 16 | Show the same weighted operation in two mechanisms while keeping interfaces visible. |
| 5 | Sparse timeline and dense/event update comparison | 7–8 | Explain why exploiting changes reduces counted work. |
| 6 | Four architectural mechanisms and notebook evidence | 5 | Establish categories without implying mutual exclusivity or universal superiority. |

All six use white backgrounds, rectangular elements, large sans-serif labels, blue/grey/muted-green colours and restrained arrows. Arrow labels identify what moves: data, state, events, optical signal or measured result. Do not use generated illustrations for exact probability charts, timings or circuit relationships. No infographic has been generated and no PowerPoint has been changed.

## Reference placement for the eventual PowerPoint

Add course reading and a source link in the notes of the relevant concept slide, rather than a long live reference sequence. Preserve original image credits for reused illustrations. Use primary sources for device capabilities and avoid unqualified “times better” figures.

- Session scope and progression: official syllabus Session 26 and Session 27.
- Course reading: Oscar Diez, *High Performance Computing: Applications and Frontiers*, Chapters 17–18; *Foundations of High-Performance Computing*, Chapter 18. Relevant course-book contents were inspected alongside the syllabus; no textbook speedup claim supplies the notebook results.
- Slide 13: Intel neuromorphic research, https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html
- Slides 14–16: Feldmann et al., *In-memory computing on a photonic platform*, https://pmc.ncbi.nlm.nih.gov/articles/PMC6377270/ . The notebook does not simulate that device.
- Slides 19–22: IBM Quantum Learning, *Superposition with Qiskit*, https://quantum.cloud.ibm.com/learning/en/modules/quantum-mechanics/superposition-with-qiskit
- Slides 8, 11, 18, 21: the notebook code, source hash and `validation_local_results.json`. Label saved author results LOCAL_CPU until replaced by actual matching SciTech evidence.
- Slide 25: the notebook’s explicit illustrative model and assumptions. No hardware timing source exists for these assumed milliseconds.

# Deliverable C · PowerPoint implementation checklist

1. **Preserve and label the source.** Keep the attached 108-slide deck unchanged as the original. Work on a new copy only when implementation is requested. Record original slide IDs/numbers before moves.
2. **Move first.** Create Backup and move the exact grouped originals listed above. Use original 76 as its divider. Quarantine unverified claims and questions rather than presenting them as approved teaching material.
3. **Reuse the identity.** Retain the cover, IE logos, original master, painting question/discussion layout from 51 and selected recap artwork from 54–55. Correct 6.2/M6.S2 labels to M6S1/Session 26 wherever content survives.
4. **Merge source material.** Assemble original 6+7+11 into new 4; 13+14+15+21 into new 6; 22+105 into new 9; 23+25 into new 13; 36+37+100 into new 14; 35+46+50 into new 24; and 54+55 into new 28. Remove surplus live duplicates after merging.
5. **Rewrite major comparisons and questions.** Rebuild original 18 as the four-mechanism map. Rewrite original 3, 31 and 51 around the specified questions/activity. Move the original answer slides to Backup and place only the new answers in notes.
6. **Add missing teaching blocks.** Create the new E1/E2 prediction/demo slides, in-memory/photonic bridge, quantum concept/interference/maturity slides, complete-cost model and five distributed MCQs. Follow the final-order table rather than appending all new material at the end.
7. **Use verified examples.** Copy data from E1–E4 records and E5 assumptions exactly. Display the evidence type on each demo. The C source stays in the notebook; show only the short snippets in these specifications. Never label saved local plots as SciTech measurements.
8. **Add diagrams in ranked order.** Implement the six priorities with editable relationships and native charts/tables where values must be exact. Retain source credits and preserve painting proportions. No decorative device image should substitute for an explanatory mechanism.
9. **Add teacher notes.** Use all five note bullets for every teaching slide, including timing, expected answers and misconceptions. MCQ notes explain every distractor. Keep answers off student-facing question slides.
10. **Check coherence.** Use the exact notebook labels E1–E5, `before_reset`, `spike`, `after_reset`, `bits`, `phase` and `overhead_ms`. Preserve scope boundaries for operation counts, CPU simulations, quantization models and illustrative times.
11. **Complete the SciTech acceptance run.** Open the raw notebook, record a real CPU job, collect matching results and review all statuses. If resources or tooling fail, retain the honest saved local demonstration and identify the specific missing validation. No current claim of cluster success is justified.
12. **Check the final teaching copy after implementation.** Verify 29 live slides, 80 minutes, five MCQs, readable diagrams, no hidden answers, no obsolete claims, a four-essentials recap and an exit bridge to Session 27. Verify the full deck visually before delivery. This checklist is a future implementation instruction; none of these PowerPoint edits has occurred.
