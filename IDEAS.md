# PLAN.md — Agentic Personal Trainer Using Computer Vision

## Project goal

Build a portfolio-ready, browser-based personal trainer that uses a pretrained pose-estimation model, deterministic exercise-form checks, and an agentic AI layer to guide workouts and explain feedback.

**Hard constraint: Do not create, collect, label, or train on a custom dataset.** The initial project must work without any dataset. Use pretrained components and rule-based exercise analysis. A public dataset or a separately pretrained classifier may be considered only as an optional extension if it can be integrated without collecting or labeling data yourself.

The project should demonstrate practical AI engineering, computer vision, agent/tool orchestration, retrieval-augmented generation (RAG), testing, and CI/CD. Do not claim capabilities or metrics that have not been implemented and tested.

## Product scope

### Core user experience

A user opens the web app, grants camera access, selects an exercise, and follows a guided session. The app estimates body pose, counts repetitions, checks a small set of interpretable form conditions, and presents concise feedback. An AI coach can answer exercise questions using retrieved reference material and invoke a limited set of safe workout tools.

### MVP exercises

1. **Squat**
   - Detect pose landmarks using a pretrained pose-estimation library such as MediaPipe Pose.
   - Calculate knee and hip angles from landmarks.
   - Count repetitions using a documented state machine (standing → descending → bottom/ascending → standing).
   - Show only rule-based cues that can be calculated reliably from available landmarks.
2. **Reverse lunge**
   - Reuse the pose-landmark pipeline.
   - Count repetitions using a simple state machine.
   - Add a small number of interpretable checks, with clear handling for low-confidence or occluded landmarks.

Do not promise medical assessment, injury prevention, or professional-grade form correction. Present feedback as general fitness guidance, not medical advice. Include a disclaimer and advise users to stop if they experience pain.

## Non-goals

- No custom dataset collection, annotation, or labeling.
- No training or fine-tuning a model as part of the required MVP.
- No claim that the system detects every form error or provides clinical/medical guidance.
- No complex multi-agent architecture unless a simple tool-calling agent proves insufficient.
- No cloud inference dependency for the core pose-estimation path if browser/local inference is practical.
- No invented accuracy, latency, user-study, or improvement metrics.

## Recommended architecture

### Front end

Use a browser-based UI that:
- Requests camera permission explicitly and explains why it is needed.
- Displays the camera preview and pose overlay.
- Shows selected exercise, rep count, set/rep target, session timer, and rest timer.
- Shows short, actionable cues with a confidence/availability state.
- Allows the user to pause, reset, and end a session.
- Displays a post-session summary using only data collected during that session.
- Keeps camera frames local to the device by default; do not save or upload video.

Use the existing repository's framework if it is viable. Do not rewrite a working project without a clear reason.

### Pose estimation

Use a suitable pretrained pose-estimation package, preferably MediaPipe Pose / Pose Landmarker, depending on compatibility with the repo and target browser.

Requirements:
- Use the package's existing pretrained model; do not train it.
- Normalize landmark coordinates and track visibility/confidence.
- Handle missing, low-confidence, and out-of-frame landmarks without producing misleading feedback.
- Keep camera processing separate from UI rendering where practical.
- Document browser and device limitations.

### Exercise analysis

Implement a small, testable rules module independent of the UI.

**Angle calculation**
- Use vector math to calculate angles at joints.
- Handle zero-length vectors and missing landmarks safely.
- Add unit tests for representative coordinates and invalid inputs.

**Rep counter**
- Use a finite-state machine with documented states and transition thresholds.
- Add hysteresis/cooldown or equivalent safeguards against duplicate counts and jitter.
- Make thresholds configurable and explain that they may need individual adjustment.
- Unit-test full rep cycles, partial reps, noisy sequences, and missing landmarks using hand-authored synthetic landmark sequences. These are software test fixtures, not a training dataset.

**Form checks**
- Start with a small number of simple, explainable rules, such as detecting a squat depth threshold or a visibly unstable/invalid landmark signal when supported by the pose output.
- Do not make a rule sound more precise than its input allows.
- Distinguish “possible form cue” from “unable to assess” when visibility is poor.
- Avoid universal claims such as “your knees must never pass your toes.”
- Keep cue text configurable and testable.

### Agentic coach

Use a local or otherwise affordable language model runtime if feasible. A small Gemma model may be used if supported by the available hardware and tool-calling approach; if reliable tool calling is not supported, implement a constrained parser or use a compatible model/runtime rather than pretending the model can call tools.

The agent should have a small, explicit set of tools, for example:
- `get_session_state`
- `start_set`
- `start_rest_timer`
- `summarize_session`
- `search_exercise_knowledge`

Requirements:
- Validate every tool argument and reject unsupported actions.
- Keep rep counting and form analysis deterministic; the LLM must not invent reps or override sensor state.
- Do not let the model control camera permissions or access arbitrary files/network resources.
- Keep coaching responses short and grounded in retrieved material where relevant.
- Provide a graceful fallback if the model is unavailable.

### RAG knowledge base

Create a small, curated reference collection from reliable, permitted-to-use exercise guidance sources. This is a knowledge base, **not a model-training dataset**.

Requirements:
- Record source name, URL, access date, and any applicable license/usage restrictions.
- Chunk documents, create embeddings using a suitable pretrained embedding model, and retrieve relevant passages.
- Show source references in answers where feasible.
- Evaluate retrieval with a small hand-written set of questions and expected relevant sources.
- Avoid medical or rehabilitation claims; use general fitness guidance only.
- If source licensing or access is unclear, use a small set of authored notes with links to reputable public references instead of copying substantial text.

### Workout flow

- Let the user select squat or reverse lunge.
- Support a simple target such as sets × reps.
- Track sets and repetitions using the deterministic exercise module.
- Offer rest periods between sets.
- Provide a session summary: completed sets/reps, elapsed time, and cues observed.
- Only suggest next-set adjustments within a small, documented range and let the user accept or reject them.
- Do not claim that a cue improved the user's form unless a repeatable measurement supports that conclusion. For the MVP, report cues shown and subsequent observations separately, with an “unable to determine” state when appropriate.

## Testing and quality

No custom dataset is required. Use **hand-authored synthetic test fixtures** for deterministic logic and scripted evaluation cases for the agent/RAG.

### Unit tests
- Joint-angle calculations, including edge cases.
- Rep-state-machine transitions and duplicate-count protection.
- Landmark confidence and missing-landmark behavior.
- Form-rule outputs and “unable to assess” conditions.
- Tool schemas, argument validation, and safe rejection of unsupported actions.
- Session state, set completion, rest timer state, and summary calculations.

### Integration tests
- Pose output can be passed into the exercise-analysis module.
- Exercise events update session state correctly.
- Agent tool calls use validated application state.
- RAG retrieval returns relevant source passages for authored test questions.
- App remains usable if the camera, model runtime, or retrieval service fails.

### Manual acceptance checks
- Camera permission granted and denied.
- Different lighting and camera positions.
- Landmark occlusion and partial body visibility.
- Several complete and incomplete squat/lunge cycles.
- Browser refresh and session reset.
- Keyboard access and readable feedback.
- No video uploaded or persisted unexpectedly.

Synthetic tests validate program logic, not real-world pose-estimation accuracy. If a public benchmark or public dataset is used later, document its source and limits and keep it optional.

## MLOps / delivery approach

Because the MVP does not train a model, **do not force a training pipeline or fabricate MLflow experiments**.

Implement meaningful software/ML delivery practices:
- Pin and document dependencies.
- Use GitHub Actions to run formatting/linting, type checks where applicable, and pytest on pushes and pull requests.
- Add smoke tests for the core analysis and session modules.
- Track model/package names, versions, licenses, and configuration in documentation.
- Version application rules and configuration in Git.
- If MLflow adds genuine value for evaluating alternative configurations or tracking an optional externally pretrained model, integrate it as an extension. Do not claim model training or model-registry work unless it exists.
- Add a reproducible setup guide and a clear model/runtime fallback.

## Suggested technology choices

Prefer technologies already present in the repository. If starting fresh, consider:
- Python for backend or application logic where needed.
- TypeScript/JavaScript for a browser-first UI.
- MediaPipe for pretrained pose estimation.
- NumPy for geometry where useful.
- A local LLM runtime and Gemma-compatible model if hardware/runtime support is adequate.
- A lightweight vector store or local retrieval implementation for RAG.
- pytest for tests.
- GitHub Actions for CI.
- Optional MLflow only if there is a real experiment-tracking use case.

Do not add Pandas, PyTorch, MLflow, or other technologies solely to list them on a résumé. Only include a technology when the implementation genuinely uses it.

## Implementation phases

Work one phase at a time. At the end of each phase, run the relevant checks, report actual results, and fix failures before proceeding. Do not claim tests passed unless they were run.

### Phase 0 — Inspect repository and confirm design
- Read `PLAN.md`, `CLAUDE.md` if present, and existing project documentation.
- Inspect the repository, dependencies, scripts, and current application state.
- Identify the smallest viable implementation that fits the current stack.
- Confirm that no custom dataset or training step is needed.
- Propose a concise file-level implementation plan before major edits.

**Gate:** agree on the stack and commands based on repository evidence.

### Phase 1 — App skeleton and local setup
- Preserve useful existing code.
- Create or repair the app shell, configuration, and setup instructions.
- Add dependency and environment documentation.
- Add a health/smoke check appropriate to the stack.

**Gate:** app starts using documented commands.

### Phase 2 — Pose estimation and landmark handling
- Integrate the pretrained pose model.
- Implement camera permission and failure states.
- Display landmarks/overlay.
- Add confidence and missing-landmark handling.
- Document the local-processing/privacy behavior.

**Gate:** pose output is visible and invalid inputs are handled safely.

### Phase 3 — Exercise geometry and rep counter
- Implement angle helpers.
- Implement the squat state machine.
- Add synthetic unit fixtures and tests.
- Add configurable thresholds and explain them in documentation.

**Gate:** automated tests cover full, partial, noisy, and invalid cycles.

### Phase 4 — Form cues and second exercise
- Implement conservative squat cues.
- Add reverse-lunge counting and a small set of checks.
- Add “unable to assess” behavior for poor visibility.
- Test the rules independently from the camera/UI.

**Gate:** rules are deterministic, documented, and tested.

### Phase 5 — Guided workout sessions
- Add exercise selection, sets/reps target, rest timer, pause/reset, and session summary.
- Ensure counts and summaries come from application state rather than LLM output.
- Add session logic tests.

**Gate:** a user can complete a simple end-to-end workout flow.

### Phase 6 — Knowledge base and RAG
- Add a small permitted-to-use knowledge collection with source metadata.
- Implement retrieval and source-aware responses.
- Create hand-authored RAG evaluation questions.
- Add tests for no-result and irrelevant-result cases.

**Gate:** test questions retrieve useful sources and answers do not invent citations.

### Phase 7 — Agent tools and coaching
- Integrate the chosen LLM/runtime if feasible.
- Add explicit tool schemas, argument validation, and safe fallbacks.
- Ground exercise explanations in RAG.
- Keep sensor analysis deterministic.
- Test invalid tool requests and runtime failure.

**Gate:** the coach can safely use a small set of tools and remains useful when the model is unavailable.

### Phase 8 — CI, documentation, and polish
- Add GitHub Actions for relevant automated checks.
- Pin dependencies and document model/package licenses.
- Add setup, architecture, limitations, privacy, and troubleshooting docs.
- Add screenshots or a short demo only after the app works.
- Run all checks and report results accurately.

**Gate:** a fresh setup follows the README and CI passes.

## Approximate schedule

A realistic target is **12–17 focused working days** with coding assistance, assuming the existing repository is usable and local model integration does not become a major blocker. The LLM/runtime and browser camera integration are the largest uncertainties. If time runs short, finish pose estimation, deterministic squat counting, a guided session, tests, and CI before adding the second exercise or agentic RAG.

## Acceptance criteria

The MVP is complete when:
- The app can use a pretrained pose model without custom training data.
- Squat reps are counted by a tested deterministic state machine.
- At least a few conservative form cues work, including a clear “unable to assess” path.
- A guided session tracks sets, reps, and rest.
- The agent has a small validated tool set, or a documented fallback if the selected runtime cannot support reliable tool calling.
- RAG uses a documented, permitted-to-use knowledge source and passes authored retrieval checks.
- CI runs automated checks.
- The README documents setup, architecture, limitations, privacy, and the fact that no custom dataset/model training was performed.
- Résumé claims describe only features and technologies actually implemented and tested.

## Instructions for Claude Code

1. Read this entire `PLAN.md` and `CLAUDE.md` if present before changing files.
2. Inspect the repository before deciding on the framework or replacing existing code.
3. Treat “no custom dataset” as a hard requirement: do not collect, label, generate, or train on an exercise dataset.
4. Use pretrained pose estimation and deterministic rules for the required MVP. Hand-authored synthetic fixtures are allowed only for testing program logic.
5. Do not add dependencies just for résumé keywords.
6. Implement one phase at a time and respect each gate.
7. Run tests and report the actual commands and results. Be explicit about anything not tested.
8. Do not fabricate accuracy, latency, user-study, model-training, MLflow, or deployment results.
9. Keep camera/video processing local by default; never persist or upload video unless the user explicitly approves a future scope change.
10. When a design choice is uncertain, explain the trade-off and choose the simplest safe option that satisfies the MVP.
