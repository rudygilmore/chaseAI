# [AGENTS.md](http://AGENTS.md)

## Purpose and Philosophy

This repository is a research-oriented deep learning and reinforcement learning project focused on developing autonomous agents capable of playing classic console video games.

The long-term objective is to develop a reusable, automated pipeline that can accept a supported game ROM, configure an appropriate environment, train an agent, evaluate its performance, and ultimately solve simple games with minimal human intervention.

The project combines three complementary activities:

1. **Research and exploration:** Experiment with learning algorithms, model architectures, reward functions, exploration strategies, and other ideas. Develop hypotheses, inspect results, and learn from unsuccessful experiments.
2. **ML infrastructure development:** Build reusable, modular, well-tested components that support experimentation and comparison.
3. **Automated pipeline development:** Progressively integrate these components into a robust system capable of automating game setup, training, evaluation, and completion verification.

Contributors, both human and AI agents, should prioritize clarity, reproducibility, scientific rigor, and incremental progress.

Do not sacrifice experimentation and discovery in pursuit of premature architectural perfection.

---

## Core Principles

### 1. Research and experimentation

- Encourage experimentation with alternative approaches, including approaches that may ultimately fail.
- Prefer testing hypotheses through small, controlled experiments before committing to complex implementations.
- Make experimental assumptions explicit.
- Preserve interesting experimental results, including negative results, when they provide useful information.
- Avoid prematurely standardizing experimental approaches.
- Distinguish established findings from hypotheses and speculation.

Exploratory code does not need to meet every standard required of reusable infrastructure, but it should remain understandable and its limitations should be documented.

### 2. Reproducibility

- Record random seeds where applicable.
- Record experiment configurations, relevant software versions, and source code revisions.
- Keep experiment configurations and reusable source code under version control.
- Preserve sufficient information to reproduce important results.
- Distinguish training results from independent evaluation results.

Reproducibility does not require committing generated datasets, model checkpoints, or other large artifacts to Git. Use appropriate local or external artifact storage.

### 3. Incremental development

- Prefer small, reviewable changes.
- Build and validate individual components before integrating them.
- Establish simple working baselines before introducing additional complexity.
- Avoid premature optimization and unnecessary abstractions.
- Preserve existing functionality unless explicitly changing it.

### 4. Transparency

- Favor readable, explicit implementations over clever or opaque solutions.
- Document important assumptions and tradeoffs.
- Report actual test and experiment results.
- Do not conceal failures or silently change model or environment behavior.

---

## ExecPlans: Required Workflow

All non-trivial development work must follow the ExecPlan workflow defined in `PLANS.md`.

### When an ExecPlan is required

- New reusable features or architectural components.
- Changes to model architectures or learning algorithms.
- Significant training pipeline changes.
- Changes to environment interfaces or behavior.
- Significant data processing changes.
- Refactoring that affects behavior or interfaces.
- Major dependency or infrastructure changes.

### When a lightweight approach is acceptable

Small exploratory experiments, notebook investigations, visualization work, straightforward bug fixes, and documentation changes do not necessarily require a full ExecPlan.

However, exploratory work should still have a clearly stated objective and should not bypass planning requirements for significant changes to shared infrastructure.

### Minimum ExecPlan expectations

- Clear objective and motivation.
- Relevant background and existing architecture.
- Step-by-step implementation plan.
- Defined inputs and outputs.
- Validation strategy and acceptance criteria.
- Expected risks and failure handling.

### Execution rules

1. Read `PLANS.md` before starting work requiring an ExecPlan.
2. Inspect the existing codebase and relevant documentation.
3. Complete the ExecPlan before implementing the planned changes.
4. Follow the plan incrementally.
5. Update the plan when implementation reveals important differences from the original assumptions.
6. Document deviations and their rationale.

Plans should support good engineering, not become an obstacle to research.

---

## Project Architecture and Organization

Maintain separation between:

- Emulator and game integration.
- Environment wrappers and observation preprocessing.
- Agent and model architectures.
- RL algorithms and training.
- Evaluation and benchmarking.
- Experiment management and checkpointing.
- Game-solving orchestration.

Keep game-specific logic separate from reusable learning infrastructure.

Prefer well-defined interfaces that allow different algorithms, models, environments, and evaluation methods to be combined without unnecessary coupling.

### Separation of research and reusable infrastructure

Use the following conventions:

- `src/`: Reusable project code and stable interfaces.
- `experiments/`: Individual experiment definitions, configurations, hypotheses, and findings.
- `outputs/`: Generated experiment results, logs, checkpoints, and other artifacts.
- `notebooks/`: Exploration, visualization, and research analysis.
- `scripts/`: Executable entry points and utility scripts.
- `tests/`: Automated tests.
- `configs/`: Shared configuration templates and defaults.
- `docs/`: Architecture, technical documentation, and research notes.

Experimental implementations may initially live within an experiment-specific directory. Promote them into reusable infrastructure when their value and design become clearer.

Do not force every experimental idea into the core architecture.

---

## Python and Software Engineering Standards

### General

- Use the Python version specified in `pyproject.toml`.
- Follow PEP 8 conventions.
- Use type hints for public interfaces and important functions.
- Prefer explicit, readable implementations.
- Use `pathlib` for filesystem operations.
- Avoid unnecessary global state and hidden side effects.
- Document nontrivial classes, functions, and interfaces.

### Formatting and linting

Use Ruff for Python linting and formatting unless the project explicitly establishes another toolchain.

Avoid maintaining redundant formatting and linting tools.

Target a consistent maximum line length of 100 characters unless project configuration specifies otherwise.

### Dependencies

- Declare project dependencies in `pyproject.toml`.
- Avoid maintaining parallel dependency specifications.
- Prefer established libraries over unnecessary custom implementations.
- Explain the rationale for significant new dependencies.
- Avoid hidden dependencies and unjustified external services or network calls.

---

## Deep Learning and Reinforcement Learning Standards

### Model development

- Separate model architecture from training and evaluation logic.
- Prefer modular PyTorch implementations.
- Make model configuration explicit.
- Validate tensor dimensions, types, and numerical behavior.
- Handle device selection explicitly.
- Avoid unnecessary tensor copies and device transfers.

### Reinforcement learning

Pay particular attention to the correctness of:

- State and observation representations.
- Action-space definitions and mappings.
- Reward calculations and shaping.
- Episode termination and truncation.
- Environment resets.
- Experience collection and replay.
- Training targets and optimization.
- Exploration and evaluation policies.

Distinguish native game rewards from custom reward shaping.

Avoid introducing privileged information into the agent's observation or reward calculations without explicitly documenting its intended role.

### Hardware compatibility

The initial target environment is an Apple Silicon Mac.

Support PyTorch MPS where available, with CPU fallback.

Do not assume CUDA availability.

Keep hardware selection configurable and centralize device-management logic.

Validate MPS compatibility rather than assuming every PyTorch operation is supported identically across devices.

Profile performance before introducing hardware-specific optimizations.

---

## Experiment Management and Reproducibility

Experiments are a central part of this project and should be treated as first-class development artifacts.

Each meaningful experiment should document:

- Experiment identifier and objective.
- Hypothesis or research question.
- Environment and game configuration.
- Algorithm and model architecture.
- Hyperparameters.
- Random seeds.
- Software and dependency versions.
- Training budget and evaluation methodology.
- Results and conclusions.

Use configuration-driven experiments rather than hardcoding parameters in training scripts.

Where practical, make experiments executable through command-line interfaces and configuration files.

### Experiment tracking

Track relevant metrics, including:

- Episode reward and return.
- Episode length.
- Training loss.
- Exploration rate.
- Environment steps and training steps.
- Training throughput.
- Evaluation performance.
- Game progress.
- Deaths, failures, and completion rates where observable.

Preserve configuration and metadata alongside generated results.

Use lightweight experiment tracking initially, but maintain interfaces that allow more sophisticated tracking tools to be introduced later.

### Checkpoints and artifacts

Support saving and restoring:

- Model parameters.
- Optimizer state.
- Training progress.
- Relevant random number generator states.
- Experiment configuration and metadata.

Avoid overwriting existing experiment results or checkpoints without explicit authorization.

---

## Testing and Validation

Testing should be appropriate to the component being developed.

### Unit tests

Use fast, deterministic tests wherever possible.

Prioritize:

- Environment wrappers and transformations.
- Tensor operations and mathematical utilities.
- Model input/output shapes.
- Replay buffers.
- Reward calculations.
- Configuration validation.
- Checkpoint loading and saving.

### Integration tests

Test interactions between major components, including:

- Emulator initialization.
- Environment reset and stepping.
- Observation and action interfaces.
- Training and checkpointing.
- Model loading and evaluation.

### Training and evaluation validation

RL training is inherently stochastic. Do not treat a single training run as conclusive evidence of algorithmic correctness or effectiveness.

Where practical:

- Use controlled seeds.
- Compare against simple baselines.
- Evaluate across multiple episodes and seeds.
- Track learning curves and variability.
- Separate training and evaluation environments.
- Compare changes against established baselines.

Use short smoke tests before launching expensive training runs.

---

## Visualization and Research Analysis

Visualization and analysis are essential parts of this project, not optional presentation features.

Encourage the use of notebooks, plotting libraries, emulator recordings, and other visualization tools to understand model behavior and diagnose learning problems.

Useful visualizations include:

- Training and evaluation learning curves.
- Reward and episode-length distributions.
- Game progress over time.
- Action distributions and exploration behavior.
- Loss curves and optimization diagnostics.
- Observation and preprocessing visualizations.
- Emulator frame sequences and gameplay recordings.
- Comparisons between agents and experiments.

Prefer visualizations that help answer a specific research question or diagnose a specific problem.

Notebooks may remain permanent research artifacts when they provide useful analysis, visualization, or documentation.

Do not require all notebook work to be migrated into production scripts.

Reusable functionality developed in notebooks should be promoted into tested modules when appropriate.

---

## Performance and Resource Management

- Measure performance before optimizing.
- Track environment steps per second and training throughput.
- Identify whether bottlenecks originate in emulation, preprocessing, experience collection, or neural network optimization.
- Monitor memory usage, particularly on unified-memory hardware.
- Avoid expensive training runs when a short smoke test can validate the implementation.
- Make training budgets and computational requirements explicit.

Prefer improvements that meaningfully increase experimental throughput or reliability.

---

## Git Workflow and File Safety

### Branching

- Keep `main` stable.
- Use feature branches: `feature/<short-description>`.
- Keep changes focused and reviewable.

### Commits

- Use descriptive commit messages.
- Avoid large, unexplained commits.
- Do not commit generated artifacts or temporary files.

### Never commit

- API keys or credentials.
- `.env` files.
- Private datasets.
- Large binary artifacts without an explicit reason and appropriate storage strategy.
- Generated checkpoints, logs, and experiment outputs.

Use `.gitignore` for local data, checkpoints, outputs, virtual environments, and temporary files.

Use `.env.example` when environment-variable configuration needs to be documented.

Do not discard uncommitted user changes, rewrite shared Git history, or perform destructive operations without explicit authorization.

---

## Notebooks and Exploratory Development

Notebooks are encouraged for:

- Initial research and exploration.
- Data inspection.
- Experiment analysis.
- Visualization.
- Debugging model and environment behavior.
- Comparing algorithms and results.

Keep exploratory notebooks separate from reusable application logic.

Avoid relying on notebook execution order, hidden state, or manual steps for repeatable training workflows.

Document important conclusions and preserve the configurations and code needed to reproduce meaningful findings.

---

## AI Agent-Specific Rules

AI agents contributing to this repository must:

1. Read `AGENTS.md` and relevant project documentation.
2. Read `PLANS.md` before undertaking work requiring an ExecPlan.
3. Follow the applicable planning and implementation requirements.
4. Inspect existing code before proposing substantial changes.
5. Explain important technical decisions and tradeoffs.
6. Avoid introducing unnecessary dependencies or external services.
7. Write and execute appropriate tests.
8. Report actual execution and test results.
9. Identify assumptions, limitations, and unresolved problems.
10. Avoid destructive changes to user data, checkpoints, and experiment results.

### Autonomous implementation

When given a sufficiently defined task, agents should aim to complete the implementation rather than merely suggest code.

Agents should use available tools to:

- Inspect the repository.
- Implement changes.
- Run tests and smoke tests.
- Launch appropriately scoped experiments when authorized.
- Inspect logs and results.
- Diagnose failures.
- Update documentation.

Do not claim that an experiment succeeded or a feature works without appropriate verification.

When an experiment fails, report the failure and preserve useful diagnostic information rather than silently changing the methodology.

---

## Anti-Patterns

Avoid:

- Silent changes to model or environment behavior.
- Mixing experimental implementations with stable infrastructure without clear separation.
- Treating a single successful training run as proof of general effectiveness.
- Overfitting conclusions to individual seeds or episodes.
- Premature optimization.
- Unnecessary architectural complexity.
- Hardcoded paths and machine-specific assumptions.
- Untracked changes to experiment configurations.
- Destructive operations on checkpoints or research artifacts.
- Monolithic commits.
- Skipping required ExecPlans.

---

## Guiding Principle

The goal is to build a system that is both a productive research environment and a reliable ML engineering platform.

Favor experimentation when exploring new ideas. Favor rigor when evaluating results. Favor modularity and testing when developing reusable infrastructure. Favor automation when integrating proven components into the eventual game-solving pipeline.

Maintain enough structure to make progress reproducible without making the process of discovery unnecessarily rigid.
