# Pluggable Problem Architecture — Design

Date: 2026-07-23
Branch: `pluggable-problem-architecture`
Status: approved for planning (autonomous session; decisions integrated from two independent expert reviews — deep-reasoner and Codex — which converged on the same target architecture)

## 1. Goal

Make the PPO harness extensible along three independent axes so a future user can add a
new problem without editing shared harness code:

- **(a) Material / problem domain** — a new parameter/composition space and episode shape.
- **(b) Objective** — a new reward/termination/ranking rule (e.g. "maximize property above threshold").
- **(c) Prediction model backend** — physics simulator (verilogae EEHEMT), XGBoost committee
  package, or a future ANN surrogate.

Acceptance criterion (from both expert reviews): **a third problem can be registered from
new code only** — one problem package plus one `register()` call, zero edits to
`train_ppo.py`, `training/ppo_common.py`, callback plumbing, evaluation worker plumbing,
or checkpoint construction.

## 2. Why (current-state findings)

Both independent reviews agreed:

- Dispatch is hardcoded in 4 sites across 2 shared files: `train_ppo.py:22-27`
  (`select_training_module` if/elif), `train_ppo.py:35` and `training/ppo_common.py:10`
  (duplicated `choices=["hardness", "eehemt"]`), `train_ppo.py:64-69` (W&B project if/elif).
- There is no objective abstraction: reward/termination live inline in each `env.step()`;
  the ranked checkpoint metric name is a bare string that must agree across the env info
  keys, the callback (`utils/hardness_callbacks.py:38`), the checkpoint config
  (`training/hardness_ppo.py:109`), and the eval artifact writer — with no enforcement.
  Symptom: `success_rate_650` metric name vs `HARDNESS_THRESHOLD=600` today.
- The backend seam is asymmetric: `MaterialHardnessEnv` injects `inference_model_cls`
  (tests stub through it) but consumes hardcoded DataFrame columns
  (`"Predicted hardness"`, `env/material_hardness_env.py:75`); `EEHEMTEnv` constructs
  `EEHEMTSimulator.from_va_file(...)` internally (`env/eehemt_env.py:107`) — not injectable.
- A third problem today costs an estimated 4–7 new files + edits to 2–4 shared files,
  ~300–500 lines of copied integration plumbing before any domain logic.

## 3. Approaches considered

- **A. Registry only.** Fix dispatch, leave objective/backend coupling as is.
  Cheap, but fails goal axes (b) and (c). Rejected.
- **B. Fully generic environment.** One generic `gym.Env` assembled from
  adapter + backend + objective; both existing envs become configurations of it.
  Maximum uniformity, but forces the multi-step, solver-coupled EEHEMT episode and the
  single-step hardness search into one candidate model — the over-generalization risk
  called out explicitly in review. Rejected for this pass.
- **C. Staged hybrid (chosen).** Registry + typed backend protocol + objective strategy
  + deduplicated PPO assembly + parameterized callback/eval plumbing. The two envs stay
  concrete `gym.Env` classes that *delegate* to the new abstractions. Facades and metric
  strings preserved verbatim. Delivers all three axes with the lowest semantic-drift risk.

## 4. Target architecture (approach C)

### 4.1 Problem registry — `problems/`

New top-level package (additive layer; existing layer boundaries untouched):

```
problems/
  __init__.py      # imports hardness, eehemt so they self-register
  registry.py      # ProblemSpec + register() / get() / names()
  hardness.py      # builds ProblemSpec from existing training/hardness_ppo.py parts
  eehemt.py        # builds ProblemSpec from existing training/eehemt_ppo.py parts
```

```python
@dataclass(frozen=True)
class ProblemSpec:
    name: str                       # CLI name, e.g. "hardness"
    add_env_args: Callable          # argparse group for problem-specific flags
    build_ppo_config: Callable      # (args) -> PPOConfig, delegates to generic builder
    build_checkpoint_config: Callable
    wandb_project: str              # replaces _wandb_project_name if/elif
    checkpoint_metric: str          # e.g. "env_runners/max_predicted_hardness" (verbatim)
    checkpoint_order: Literal["min", "max"]
```

- `train_ppo.py` resolves `--env` via `registry.get(name)`; both `choices=` lists become
  `registry.names()`; the W&B if/elif becomes `spec.wandb_project`.
- `select_training_module` is kept as a thin compatibility facade (tests import it).
- Default stays `hardness`; `--env eehemt|hardness` behave identically to today.
- Specs hold import-safe callables/strings only (Ray-serializable; live objects are
  constructed inside env/config builders, mirroring how `inference_model_cls` already
  travels through env_config).

### 4.2 Prediction backend protocol — `env/backends.py`

```python
@dataclass(frozen=True)
class PredictionResult:
    values: Mapping[str, float]         # target name -> prediction
    uncertainties: Mapping[str, float]  # target name -> std (diagnostic only)

class PredictionBackend(Protocol):
    def predict(self, features: Mapping[str, float]) -> PredictionResult: ...
    def close(self) -> None: ...        # default no-op mixin provided
```

- `CommitteePackageBackend` wraps the existing `InferenceModel`; it derives the target
  name from the package's declared targets (`env/inference_engine.py:95-98`) instead of
  the consumer hardcoding the `"Predicted hardness"` column literal.
- `env.InferenceModel` / `env.predict` public exports stay unchanged (tested surface).
- `MaterialHardnessEnv` accepts a new `backend` / `backend_factory` env-config entry;
  the legacy `inference_model_cls` path keeps working during migration (dual-path,
  compared for identical reward/info output in tests).
- `EEHEMTEnv` gains an injectable simulator factory defaulting to today's
  `EEHEMTSimulator.from_va_file(...)` construction — closing the asymmetry and giving
  EEHEMT a test seam for the first time. This touches a stable layer boundary → ADR.
- A future ANN backend implements `PredictionBackend` directly; it does not need the
  committee ZIP format, and must not inherit the verilogae Linux/py3.11 toolchain.

### 4.3 Objective strategies — `env/objectives.py`

```python
class ObjectiveOutcome(NamedTuple):
    reward: float
    terminated: bool
    success: bool
    metrics: dict[str, float]   # finite scalars for info/callback consumption

class Objective(Protocol):
    ranked_metric: str          # single source of truth for the checkpoint metric name
    ranked_order: Literal["min", "max"]
    def evaluate(self, observation_context) -> ObjectiveOutcome: ...
```

Two built-ins reproduce today's numerics **byte-for-byte** (locked by ADR 0001 and the
existing test tables):

- `ThresholdMaximizeObjective` — `(predicted - threshold) / scale`, clipped;
  `terminated=True` (single-step); success when `predicted >= threshold`;
  `ranked_metric="env_runners/max_predicted_hardness"`, order `max`.
- `NRMSEMinimizeObjective` — `clip(-log10((NRMSE/100) + EPSILON), REWARD_MIN, REWARD_MAX)`;
  success/termination only when `NRMSE < NRMSE_THRESHOLD`;
  `ranked_metric="env_runners/min_nrmse"`, order `min`. Solver non-convergence penalty,
  truncation, and episode-best tracking (ignoring non-converged candidates) remain in
  `EEHEMTEnv` — they are episode/solver orchestration, not objective math — but the
  objective owns the reward transform and the success comparison.

Envs delegate to their objective; `ProblemSpec.checkpoint_metric` is sourced from the
objective instance so the string exists in exactly one place. Uncertainty is available to
`evaluate()` for metrics but built-ins never let it into reward (guardrail).

### 4.4 Generic PPO assembly — `training/ppo_common.py`

Extract the duplicated `.env_runners()/.training()/.learners()/.evaluation()` chain
(compare `training/eehemt_ppo.py:87` with `training/hardness_ppo.py:68`) into one
`build_base_ppo_config(args, *, env_cls, env_config, callback_cls, eval_fn)` helper.
Both problem modules become thin: parse their args, build env config, call the base
builder, attach their checkpoint config. Public functions and signatures of
`training/eehemt_ppo.py` / `training/hardness_ppo.py` survive as the problem-module API.

### 4.5 Callback / evaluation plumbing

Extract the RLlib-new-API-stack mechanics shared by both custom eval functions
(worker enumeration, `get_infos` episode extraction with fallbacks, metric aggregation
key path, per-algorithm evaluation index attr) into a shared helper module
(`evaluation/rllib_plumbing.py`). Keep **two thin artifact renderers**
(I-V curves; hardness compositions) — renderers are genuinely domain-specific.
Callbacks become parameterized by the metric names the objective/spec provides;
emitted metric names stay verbatim (`max_predicted_hardness`, `min_nrmse`,
`success_rate_650` continues to be emitted for dashboard compatibility, with a
threshold-agnostic `success_rate` added alongside; the naming wart and its deprecation
path are recorded in the ADR).

### 4.6 Acceptance test + docs

- `tests/test_problem_registry_extension.py`: registers a deterministic **toy problem**
  (stub backend + `ThresholdMaximizeObjective` + minimal env) entirely from test code and
  asserts: CLI `choices` include it, dispatch resolves it, PPO config builds, checkpoint
  config carries its metric — **without any shared-file edit**. This is the executable
  proof of the extension model.
- `docs/how-to-add-a-problem.md`: step-by-step guide for future users (problem package,
  backend adapter, objective choice, registration, .env keys, model artifact placement).
- CONTEXT.md gains the new ubiquitous terms: Prediction Backend, Objective, Problem Spec.
- New ADR in `docs/adr/` + `.codebase-memory/adr.md`: problem registry, backend protocol,
  EEHEMT injection seam, metric-name compatibility policy.

### 4.7 Bundled hygiene fixes (pre-existing, verified identical on main)

- `mypy.ini`: inline comments break ini boolean parsing ("Not a boolean: True") — move
  comments to their own lines; document `uv run mypy .` as the canonical invocation so the
  project venv is used. Fix the `utils/dim_reduce.py` duplicate-module-name resolution.
- `ruff`: fix the 7 outstanding errors (notebook E402s via per-file ignore, unused
  import/variable in `env/file_process.py`, E402 in `scripts/run_model_inference.py`);
  migrate deprecated top-level `ignore` to `lint.ignore` in `pyproject.toml`.

## 5. Hard constraints (must hold at every migration step)

1. EEHEMT objective stays NRMSE-in-linear-current with the exact reward transform,
   strict `<` termination, `env_runners/min_nrmse` ranking; arcsinh-Huber stays
   diagnostic-only (ADR 0001).
2. Hardness action space stays exactly six tunable fractions (Al, Cr, Mn, Fe, Co, Ni),
   each in [0.05, 0.35], summing to 1.0; Cu/Mo fixed at 0; uncertainty never enters reward.
3. Checkpoint metric strings preserved verbatim (restore-path and Tune compatibility).
4. CLI backward compatible: `--env eehemt|hardness`, default `hardness`, all existing
   flags and env-var fallbacks unchanged; existing `.env` files keep working.
5. `OBSERVATION_FILTER=NoFilter` stays default; observations stay legal finite-bound
   float32 Gymnasium Boxes in all reset modes.
6. Full test suite green after every phase; objective extraction verified by
   numerical-identity assertions against the existing reward test tables.
7. Layer-boundary changes (EEHEMT injection seam, eval plumbing extraction) recorded via ADR.

## 6. Migration phases (each = separately green, separately committed)

- **P0** Characterization tests for gaps (CLI choices/default, NoFilter default, emitted
  metric names, artifact schemas) + hygiene fixes (mypy.ini, ruff). Independent of the rest.
- **P1** `problems/` registry; `train_ppo.py` + `ppo_common.py` consult it;
  `select_training_module` facade preserved.
- **P2** Generic PPO builder; both training modules delegate.
- **P3** `env/backends.py`; committee adapter; hardness dual-path injection;
  EEHEMT simulator factory seam (+ ADR).
- **P4** `env/objectives.py`; both envs delegate reward/termination; numerical-identity tests.
- **P5** Shared eval/callback plumbing module; thin renderers; metric names verbatim.
- **P6** Toy-problem acceptance test; `docs/how-to-add-a-problem.md`; CONTEXT.md terms;
  ADR entries; final full verification (pytest, ruff, mypy).

## 7. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Numerical drift in objective extraction | Identity tests against existing reward tables before deleting inline code (P4 gate) |
| Episode-best semantics lost | Episode-best/solver logic stays in `EEHEMTEnv`; only reward math and success comparison move |
| Checkpoint/restore breakage | Metric strings sourced from objective but asserted verbatim in characterization tests |
| RLlib new-API-stack churn | Plumbing extraction is mechanical relocation, not rewrite; existing eval tests must pass unchanged |
| Ray worker serialization | Specs carry classes/callables + kwargs (same pattern as today's `inference_model_cls`) |
| EEHEMT import-time global state (`env/eehemt_env.py:21-42`) | Out of scope to remove this pass; injection seam is additive and default-preserving |
| Over-generalization | Envs stay concrete; no generic env class in this pass (approach B explicitly deferred) |

## 8. Out of scope

YAML/TOML config layer; entry-point/plugin auto-discovery; renaming or removing existing
metrics; any change to reward semantics or action spaces; MeanStdFilter; removing EEHEMT
module-level parameter loading; a real (non-toy) third problem.
