# Agentic Personal Trainer — Implementation Plan

> **Purpose:** Build a portfolio-ready, browser-based personal trainer that combines computer vision, a constrained agent, retrieval-augmented generation (RAG), guided exercise sessions, evaluation, and an automated ML quality gate.
>
> **Planning estimate:** 16 working days. This is a target, not a promise. Do not skip acceptance criteria to preserve the schedule.
>
> **Execution rule:** Implement one phase at a time. Read the repository before changing it, keep changes small, run the relevant tests, report results, and stop at each phase gate for review.

---

## 1. Product definition

### 1.1 User experience

A user opens the web app, grants camera access, selects a supported exercise or a predefined session, and receives:

- Pose-based feedback and repetition counts.
- Supported form flags and concise corrective cues.
- Exercise questions answered using the curated knowledge base.
- Guided sessions with set targets, rest periods, and a session summary.
- A clear indication when tracking confidence is insufficient.

The initial supported exercises are:

1. **Squat:** MediaPipe pose landmarks, deterministic repetition state machine, and a trained PyTorch form classifier.
2. **Reverse lunge (default second exercise):** deterministic repetition logic and rule-based form checks. Do not train a second classifier in the initial scope.

The language model is not responsible for frame-by-frame pose estimation, rep counting, or making unchecked changes to the session.

### 1.2 Project goals

- Build a working end-to-end demo, not just isolated notebooks.
- Keep vision, exercise rules, session state, agent tools, and evaluation independently testable.
- Measure model performance and agent advice quality using documented test sets.
- Make CI enforce tests and explicit model-quality thresholds.
- Make resume claims only after the corresponding functionality is implemented and verified.

### 1.3 Out of scope for the first release

- Medical diagnosis, injury diagnosis, rehabilitation advice, or guarantees that a cue will correct a user's movement.
- Free-form workout generation by the LLM.
- A trained classifier for the second exercise.
- Production-scale multi-user hosting, authentication, user analytics, or persistent cloud accounts.
- Drift detection or claims of production readiness.
- ONNX/WebGPU optimization unless core requirements are complete and time remains.

---

## 2. Non-negotiable engineering rules

1. **Inspect before editing.** At the start of each phase, inspect the repository, `CLAUDE.md`, dependencies, and existing tests. Do not overwrite working code without explaining why.
2. **Plan before implementing.** State the files to create/change, interfaces, risks, and tests. Then implement the smallest coherent increment.
3. **No invented results.** Never fabricate metrics, datasets, users, latency, accuracy, or successful correction outcomes.
4. **Deterministic core.** Rep counting, session transitions, rule checks, and safety constraints must be deterministic Python code with tests. The LLM may interpret requests and explain approved information, but must not override these controls.
5. **Explicit interfaces.** Keep browser I/O, pose estimation, feature extraction, model inference, exercise logic, session management, retrieval, agent tools, and evaluation in separate modules.
6. **Reproducibility.** Fix random seeds where appropriate, record dependency versions, save dataset split identifiers, and document training and evaluation commands.
7. **Fail safely.** Low-confidence or missing landmarks must not produce confident form claims or silently increment reps. Ambiguous correction outcomes must not be counted as successful.
8. **Privacy by default.** Process camera frames locally where practical. Do not persist raw video by default. If examples are recorded for training, make that an explicit local workflow with consent and documented storage.
9. **Test every phase.** Add unit tests with each component. Use mocks for camera, model, timer, and LLM dependencies where appropriate.
10. **Do not add dependencies casually.** Explain why a new package is needed and prefer the standard library or existing dependencies when reasonable.
11. **Optional libraries are optional.** Use NumPy and Pandas only where they materially help feature engineering, data preparation, or analysis. Do not add them just for resume keywords.
12. **No false CI claims.** CI validates committed artifacts and runs tests; it does not retrain a model unless a reproducible training job is deliberately implemented.
13. **No premature completion.** A phase is complete only when its acceptance criteria pass and its limitations are recorded.

---

## 3. Proposed architecture

Adapt names to the repository if a different structure already exists, but preserve the boundaries.

```text
agentic-personal-trainer/
├── PLAN.md
├── CLAUDE.md
├── README.md
├── BENCHMARKS.md
├── MODEL_CARD.md
├── pyproject.toml
├── src/
│   └── personal_trainer/
│       ├── app/                 # API / application wiring
│       ├── vision/              # MediaPipe adapter and landmark validation
│       ├── features/            # Angles, normalization, feature schema
│       ├── models/              # PyTorch model, loading, inference
│       ├── exercises/            # Exercise interfaces and implementations
│       │   ├── base.py
│       │   ├── squat.py
│       │   └── reverse_lunge.py
│       ├── sessions/             # Session state machine and summaries
│       ├── agent/                # Gemma adapter, validated tools, safety policy
│       ├── retrieval/             # Curated knowledge base and RAG
│       ├── evaluation/            # Metrics, scenario runner, reports
│       └── logging_config.py
├── frontend/                     # Browser camera and workout UI (if separate)
├── data/
│   ├── README.md                 # Data instructions; no private/raw data in Git
│   └── examples/                 # Tiny synthetic fixtures only, if needed
├── models/                       # Documented local artifacts; large files via LFS/releases if needed
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
└── .github/
    └── workflows/
        └── ci.yml
```

Do not create every file or directory at once. Add each module when its phase begins. Keep secrets, personal video, and large datasets out of Git.

### 3.1 Data flow

```text
Browser camera
    -> pose estimation
    -> landmark/confidence validation
    -> feature extraction
    -> exercise-specific rep state machine + form checks
    -> structured events (rep, form flag, cue, tracking lost)
    -> session state machine and UI

User question
    -> validated agent tool / RAG retrieval
    -> on-device Gemma response
    -> response validation and UI

Training data
    -> split by recording/session/person where feasible
    -> train candidate model
    -> held-out evaluation
    -> versioned model artifact + metadata
    -> CI tests and explicit quality gates
```

### 3.2 Architecture decisions to settle in Phase 0

Record decisions in `CLAUDE.md` or a short architecture decision record:

- Frontend framework and browser-to-Python communication method.
- Whether the camera/MediaPipe inference runs in the browser or Python backend. Choose one coherent initial path; avoid duplicate pose pipelines.
- Supported Gemma model and inference runtime, including browser/hardware requirements.
- Model artifact storage strategy and how CI obtains the committed/versioned artifact.
- Knowledge-base format and retrieval implementation.
- Supported browser, operating system, camera view, and expected input conditions.

Do not claim fully on-device inference until the selected Gemma runtime actually runs on the target device without a remote inference service.

---

## 4. Phase plan and acceptance criteria

The estimates below total approximately 16 working days. If a dependency or model runtime proves unsuitable, report the blocker and propose the smallest scope adjustment rather than silently substituting technology.

### Phase 0 — Repository and specification (Day 1)

**Tasks**
- Inspect repository and existing implementation.
- Choose the initial app architecture and supported runtime.
- Create project configuration, formatting/linting, and pytest setup.
- Write `CLAUDE.md` with architecture, commands, coding rules, and phase-gate requirements.
- Add a basic app entry point and smoke test.
- Document how to acquire, label, and split exercise data.

**Acceptance criteria**
- A fresh environment can install dependencies using documented commands.
- The app's health/smoke test and pytest pass.
- `CLAUDE.md` and `README.md` contain the actual chosen commands and architecture.
- No secrets, raw personal videos, or large untracked artifacts are committed.

**Gate:** Review architecture and runtime feasibility before implementing the vision pipeline.

### Phase 1 — Pose estimation and feature schema (Days 2–3)

**Tasks**
- Implement a thin MediaPipe adapter.
- Define a versioned feature schema for normalized landmarks and selected joint angles.
- Validate landmark presence and confidence.
- Represent tracking loss explicitly.
- Add deterministic angle/normalization helpers and fixtures.
- If NumPy is useful for vectorized calculations, use it; do not add Pandas unless tabular data workflows need it.

**Acceptance criteria**
- Known synthetic landmark fixtures produce expected angles/features within a documented tolerance.
- Missing or low-confidence landmarks are rejected or marked unavailable.
- Feature names, order, units, normalization, and schema version are documented.
- Unit tests pass without requiring a physical camera.

**Gate:** Freeze the first feature schema before collecting/training data.

### Phase 2 — Squat repetition counter and classifier (Days 4–5)

**Tasks**
- Implement a deterministic squat repetition state machine with hysteresis/debouncing to avoid double counts.
- Define form classes that can be labeled consistently; do not claim a class the dataset cannot support.
- Create a documented data collection and labeling workflow.
- Split data by recording/session (and by person where feasible) to reduce leakage.
- Train a PyTorch baseline, initially an MLP on frame/landmark features.
- Try a 1D-CNN only if sequence data volume and quality justify it.
- Evaluate on a held-out set; record per-class precision/recall/F1 and confusion matrix where meaningful.
- Save model weights, feature schema version, class mapping, and training metadata.

**Acceptance criteria**
- Scripted pose sequences verify rep counting and edge cases.
- Training and evaluation commands are documented and repeatable.
- Held-out results are reported honestly; no threshold is invented after seeing test results without documenting the selection process.
- Inference rejects incompatible feature-schema versions.
- The model card records dataset size, labeling rules, split strategy, limitations, and intended use.

**Gate:** If labeled data is insufficient, keep the classifier explicitly experimental and do not make strong accuracy claims.

### Phase 3 — Browser feedback (Day 6)

**Tasks**
- Build camera permission, start/stop, tracking-status, rep-count, and form-feedback UI.
- Connect the frontend to the selected inference path.
- Show a clear unavailable/tracking-lost state instead of stale confident feedback.
- Measure pose/rep feedback latency separately from LLM latency.
- Add an integration test around the application boundary.

**Acceptance criteria**
- A user can start and stop a session and see rep/form events.
- Camera denial and temporary tracking loss are handled.
- Rep counting does not depend on the language model response.
- Any latency number in documentation has a stated environment and measurement method.

### Phase 4 — Curated exercise knowledge and RAG (Day 7)

**Tasks**
- Create a small, reviewed exercise knowledge base with source/provenance fields.
- Include supported form flags, cues, common limitations, and safe refusal/escalation language.
- Implement retrieval with a clear boundary between retrieved content and system instructions.
- Define a fixed set of scripted questions/scenarios.
- Implement a no-RAG baseline for later comparison.

**Acceptance criteria**
- Retrieval returns relevant entries for known queries.
- Unknown or unsupported questions produce a cautious response rather than fabricated exercise facts.
- Scenario set and scoring rubric are committed and versioned.
- RAG and baseline can be evaluated on the same inputs.

### Phase 5 — Gemma agent and constrained tool calling (Day 8)

**Tasks**
- Integrate the selected Gemma model/runtime after verifying hardware and browser/backend constraints.
- Define typed, allowlisted tools for current session state, rep count, form flags, exercise guidance, and summaries.
- Validate tool inputs and outputs with explicit schemas.
- Keep safety constraints and state mutations in deterministic code.
- Log structured tool-call outcomes and errors without recording unnecessary personal content.
- Test model/runtime failure and timeout behavior.

**Acceptance criteria**
- The agent can invoke only registered tools.
- Invalid arguments and unsupported tool requests are rejected safely.
- The model cannot directly set arbitrary reps, bypass rest/session transitions, or override safety rules.
- If the model is unavailable, the CV and session core continue to function where possible.
- Documentation accurately states whether inference is local, browser-based, or backend-based.

### Phase 6 — Guided squat sessions (Day 9)

**Tasks**
- Implement a deterministic session state machine with states such as ready, set-active, rest, paused/tracking-lost, completed, and aborted.
- Support a predefined target such as three sets of eight reps and a configurable rest duration.
- Start rest after the set target is reached; ensure duplicate events do not advance the session twice.
- Add accessible visual countdown and optional spoken/on-screen cues.
- Add validated tools such as `start_session`, `get_session_state`, and `get_session_summary`. Keep tool names consistent with actual implementation.
- Generate a summary from recorded events, not from model-invented counts.
- Permit only bounded, predefined adjustments and reject unsafe changes.

**Acceptance criteria**
- Unit tests cover set/repetition transitions, rest completion, pause/resume, abort, and session completion.
- Lost pose does not count as a rep or silently finish a set.
- Session summary matches the recorded event stream.
- Agent cannot change the routine outside the allowlisted options.
- A complete guided squat session works in the demo.

### Phase 7 — Second exercise: reverse lunge (Day 10)

**Tasks**
- Add a separate exercise implementation and state machine.
- Choose a camera view and rule set that can be observed reliably.
- Add only form rules that can be computed from the available landmarks; represent unavailable checks explicitly.
- Add exercise-specific cues and knowledge-base entries.
- Keep the squat classifier separate; do not imply the lunge has a trained classifier.

**Acceptance criteria**
- Scripted fixtures test rep counting, supported form rules, and missing-landmark cases.
- Exercise selection uses a fixed allowlist.
- Documentation explicitly states: trained classifier for squat; rule-based checks for reverse lunge.
- The guided-session machinery works with the selected exercise only where its requirements are supported; do not imply mixed-exercise routines unless implemented and tested.

### Phase 8 — Cue follow-through metric (Day 11)

**Tasks**
- Emit a structured cue event containing cue ID, exercise, triggering error, rep index, and timestamp/session sequence.
- Observe the next one to three valid reps for the same error.
- Define eligibility and outcome rules before calculating the metric.
- Classify outcomes as improved, persisted, or indeterminate/not enough observations.
- Report `improved / eligible cues` as the prototype follow-through rate and report indeterminate counts separately.
- Add scripted scenarios for a cleared error and a persistent error.

**Acceptance criteria**
- Tests verify the denominator, observation window, repeated cues, and indeterminate cases.
- The metric is reproducible from the event log.
- Reports explicitly state this is an observational prototype metric, not proof that the cue caused improvement.
- No correction percentage is placed on the resume until a real run produces it.

### Phase 9 — MLOps and CI/CD (Days 12–13)

**Tasks**
- Track training runs and parameters in MLflow; log relevant metrics and artifacts.
- Record the chosen model and metadata in a reproducible location.
- Implement tests for feature-schema compatibility, model loading, prediction shape, and deterministic inference expectations where applicable.
- Configure GitHub Actions for installation, lint/format checks, unit tests, and integration tests that do not require a live camera or model download.
- Add an explicit held-out model-quality gate using a documented threshold justified by the intended use and data; if no defensible threshold is yet available, make the gate report-only until one is established.
- Add a coverage threshold for the agreed test scope, and document what the coverage measure includes.
- Ensure the CI job evaluates committed/versioned artifacts; do not claim it retrains unless it actually does.
- Keep local MLflow configuration and artifact retention limitations documented.

**Acceptance criteria**
- CI runs on pull requests and the default branch.
- A deliberately failing test makes CI fail.
- A model-quality gate fails when a test fixture/artifact is below the configured threshold.
- Model version, feature schema, evaluation metrics, and artifact location are traceable.
- README documents training, logging, evaluation, and CI commands.

### Phase 10 — RAG scenario evaluation (Day 14)

**Tasks**
- Run the fixed scenarios with RAG enabled and disabled.
- Score correctness of issue identification, support from the knowledge base, relevance of the cue, and safety.
- Prefer a simple explicit rubric with examples of passing/failing answers.
- Report the number of scenarios and aggregate/per-category results.
- Review failures and document cases where RAG did not help or made results worse.

**Acceptance criteria**
- Both conditions use identical scenarios and scoring criteria.
- The report contains results, limitations, and representative failure categories.
- Any claimed improvement is calculated from actual scored outputs.
- No user-impact or safety-improvement claim is inferred from a small scripted evaluation.

### Phase 11 — End-to-end regression and safety checks (Day 15)

**Tasks**
- Run a full scripted session using mocked camera/model/LLM boundaries.
- Assert rep counts, error counts, cue events, set transitions, rest behavior, and summary.
- Test temporary tracking loss and recovery.
- Test unsafe requests such as increasing reps in response to a pain report.
- Test malformed routine selections and invalid tool arguments.
- Run all tests and fix regressions.

**Acceptance criteria**
- The scripted full-session test passes deterministically.
- Unsafe requests cannot override the deterministic safety policy.
- Failures are visible and logged without exposing unnecessary sensitive data.
- All required CI checks pass locally.

### Phase 12 — Documentation and portfolio release (Day 16)

**Tasks**
- Write setup and usage instructions from a clean environment.
- Document architecture, supported exercises, camera assumptions, runtime/hardware requirements, and limitations.
- Finalize `MODEL_CARD.md` and `BENCHMARKS.md` using measured values only.
- Capture a short demo of squat feedback, a guided session, the second exercise, and an agent/RAG interaction.
- Review dependency licenses and model/data terms.
- Consider ONNX/WebGPU only if all core acceptance criteria pass and a quick feasibility test confirms compatibility.

**Acceptance criteria**
- A reviewer can follow the README to run the demo.
- Documentation matches the implementation.
- Metrics include methodology and dataset/scenario scope.
- Known limitations and unimplemented features are clearly identified.
- No unsupported “production-ready,” “drift detection,” or user-impact claims appear.

---

## 5. Testing strategy

Use three layers:

### Unit tests
- Landmark validation and angle calculations.
- Rep-count state machines and form rules.
- Feature schema validation and model input shape.
- Session transitions and rest timer behavior using a controllable clock or injected timer.
- Tool schemas, allowlists, safety rules, and correction metric calculation.
- Retrieval and scenario scoring.

### Integration tests
- Pose/feature/model/exercise boundaries using recorded or synthetic fixtures.
- Agent tool dispatch with a fake model.
- Session manager receiving exercise events.
- Model artifact loading and feature-version checks.
- Application API/UI boundary without requiring a physical camera.

### Evaluation tests
- Held-out squat classification metrics.
- Rep-count accuracy on scripted/annotated sequences.
- RAG versus no-RAG scenario scores.
- Cue follow-through metric with explicit eligibility rules.
- Latency measurement with hardware/runtime noted.

Never make a unit test depend on nondeterministic LLM wording. Test structured outputs, tool calls, and scoring with mocks; keep real-model evaluation in a separate, reproducible command.

---

## 6. Data and evaluation rules

- Document where data came from, how it was labeled, and what each label means.
- Split by recording/session and, where feasible, by person. Do not randomly split near-duplicate frames across train and test.
- Preserve a final held-out set for evaluation; do not repeatedly tune to it.
- Report class counts and per-class metrics, not accuracy alone when classes are imbalanced.
- Document camera angle, clothing/occlusion limitations, and supported movement variations.
- Do not upload private footage or include it in the repository by default.
- Use a versioned scenario set for agent evaluation.
- State sample sizes beside all results.
- Treat model quality, advice correctness, and observed cue follow-through as distinct metrics.

---

## 7. Logging and privacy

Use structured events where useful, such as:

- `tracking_started`, `tracking_lost`, `tracking_recovered`
- `rep_completed`, `form_error_detected`, `cue_issued`
- `session_started`, `set_completed`, `rest_started`, `session_completed`
- `agent_tool_called`, `agent_tool_rejected`, `retrieval_completed`

Each event should include only necessary fields, such as session-local sequence, exercise, event type, and relevant outcome. Avoid storing raw frames, personally identifying information, or free-form sensitive user statements by default. Do not log secrets or model credentials.

---

## 8. Definition of done

A phase is done only when all of the following are true:

- [ ] Acceptance criteria are met.
- [ ] Relevant tests pass and the command/result is reported.
- [ ] Public interfaces and data contracts are documented.
- [ ] Errors and unavailable-input cases are handled.
- [ ] New dependencies are justified.
- [ ] Known limitations are written down.
- [ ] Changes are reviewed and committed as a coherent milestone.
- [ ] No unsupported performance or impact claims have been added.

At the end of every phase, report:
1. What changed.
2. Files created or modified.
3. Commands run and test results.
4. Acceptance criteria met/not met.
5. Risks, blockers, and the next smallest task.

Do not start the next phase automatically if an acceptance criterion is failing.

---

## 9. Resume claim guardrails

Only use claims that match the shipped implementation.

Potential wording after implementation and verification:

- “Built an agentic computer vision system that counts reps, classifies squat form, and delivers real-time pose feedback, with an on-device Gemma agent using tool calling and RAG.”
- “Implemented guided exercise sessions with set and rep tracking, rest timers, and session summaries.”
- “Built a Python MLOps pipeline using pytest, MLflow experiment tracking/model versioning, and GitHub Actions quality gates.”
- “Evaluated RAG against a no-RAG baseline on **N** scripted scenarios, achieving **X** under a documented scoring rubric.”
- “Measured a **X%** prototype cue follow-through rate across **N** eligible cues.”

Do not claim user-level error reduction, causal correction, production readiness, drift detection, or real-time LLM responses without evidence. Only list Pandas/NumPy if the code genuinely uses them.

---

## 10. Instructions for Claude Code

When asked to begin or continue this plan:

1. Read `PLAN.md` and `CLAUDE.md` first.
2. Identify the current phase from the repository state; do not assume the plan is untouched.
3. Inspect relevant files and tests.
4. Summarize the smallest implementation step and expected files.
5. Implement only that step.
6. Run the most relevant tests, then broader tests if practical.
7. Report exact commands and outcomes; never say tests passed if they were not run.
8. Update checkboxes or progress notes only when there is evidence the criteria are met.
9. Ask a focused question only when a decision blocks progress; otherwise use the documented default and record the decision.
10. Stop at the phase gate and wait for review before proceeding.

**First action:** inspect the existing repository, choose and document the application/runtime architecture, then complete Phase 0. Do not generate the entire project in one pass.
