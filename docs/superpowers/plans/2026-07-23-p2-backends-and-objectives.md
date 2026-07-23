# P2: Prediction Backends + Objective Strategies — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give both problems a typed, injectable prediction-backend seam and extract reward/success math into objective strategy classes — with byte-identical numerics and every existing test untouched and green.

**Architecture:** New `env/backends.py` defines `PredictionResult` + a `PredictionBackend` protocol and a `CommitteePackageBackend` adapter that wraps `InferenceModel` via `predict_array` + declared `targets` (no DataFrame column-name string matching). `MaterialHardnessEnv` gains a backend path via a value-reader indirection so legacy `inference_model_cls` and new `prediction_backend_cls` flow through one identical `step()` body. `EEHEMTEnv_Measure_VDS` gains a `simulator_factory` injection seam (default = today's construction). New `env/objectives.py` holds `ThresholdMaximizeObjective` and `NRMSEMinimizeObjective`; the envs delegate reward/success math to them, and checkpoint-metric strings become single-sourced from the objective classes. An ADR records the layer-boundary changes.

**Tech Stack:** Python 3.11, uv, pytest, numpy, pandas, gymnasium, Ray RLlib.

**Design spec:** `docs/superpowers/specs/2026-07-23-pluggable-problem-architecture-design.md` (sections 4.2, 4.3). Two controller refinements to that spec, to be recorded in the ADR: (1) objectives own reward math + success comparison + ranked-metric identity, while episode control (termination/truncation/solver handling) stays in the envs — this avoids forcing the episodic EEHEMT and single-step hardness problems into one candidate model; (2) the hardness legacy path is preserved via value-reader indirection inside one shared `step()` flow, not a duplicated branch.

## Global Constraints

- Numerics byte-identical: hardness reward `(predicted - threshold) / scale` clipped to `[reward_min, reward_max]`, success `predicted >= threshold`; EEHEMT reward `clip(-log10((nrmse/100) + epsilon), REWARD_MIN, REWARD_MAX)`, success strictly `nrmse < NRMSE_THRESHOLD`; solver non-convergence keeps `reward = REWARD_MIN` + truncation; episode-best tracking ignores non-converged candidates.
- Checkpoint metric strings verbatim: `env_runners/min_nrmse` (order `min`), `env_runners/max_predicted_hardness` (order `max`).
- Error-message strings verbatim: `reward_scale must be positive`; `reward_min must be less than or equal to reward_max`; `Prediction output missing column: {column_name}`; `action shape must be (6,)`.
- Hardness info keys unchanged: `composition`, `predicted_hardness`, `uncertainty_hardness`, `reward_unclipped`, `is_success` (+ `raw_predicted_hardness`, `error` in the non-finite path). EEHEMT info keys unchanged.
- All existing tests stay green UNMODIFIED (129 total at plan end; per-task counts below). Verification trio after every task: `uv run pytest`, `uv run ruff check .`, `uv run mypy .` exit 0.
- Every commit message ends with the trailer line: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Work on branch `pluggable-problem-architecture`. Never commit `.claude/` or `CLAUDE.md`.
- Do not modify: `utils/callbacks.py`, `utils/hardness_callbacks.py`, `evaluation/` (Plan 3 owns those).

---

### Task 1: `env/backends.py` — PredictionResult, protocol, CommitteePackageBackend (TDD)

**Files:**
- Create: `env/backends.py`
- Create: `tests/test_backends.py`

**Interfaces:**
- Produces (used by Tasks 2 and later plans):
  - `PredictionResult(values: dict[str, float], uncertainties: dict[str, float])` — frozen dataclass.
  - `PredictionBackend` Protocol: `predict(features: Mapping[str, float]) -> PredictionResult`; `close() -> None`.
  - `CommitteePackageBackend(model_package_path, *, inference_model_cls=None)` — wraps `env.InferenceModel` (or the injected class), builds results from `predict_array` + `targets`; exposes `.targets`; `close()` forwards.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backends.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from env.backends import CommitteePackageBackend, PredictionBackend, PredictionResult


class FakeArrayModel:
    """Mimics the InferenceModel surface CommitteePackageBackend relies on."""

    def __init__(self, model_package_path):
        self.model_package_path = model_package_path
        self.targets = ["hardness"]
        self.calls = []
        self.closed = False

    def predict_array(self, input_data):
        self.calls.append(input_data)
        return np.array([[720.0]]), np.array([[12.5]]), None, None

    def close(self):
        self.closed = True


class FakeTwoTargetModel(FakeArrayModel):
    def __init__(self, model_package_path):
        super().__init__(model_package_path)
        self.targets = ["hardness", "density"]

    def predict_array(self, input_data):
        self.calls.append(input_data)
        return np.array([[700.0, 7.9]]), np.array([[10.0, 0.2]]), None, None


def test_prediction_result_is_immutable():
    result = PredictionResult(values={"hardness": 1.0}, uncertainties={})

    with pytest.raises(Exception):
        result.values = {}


def test_committee_backend_maps_targets_to_values_and_uncertainties():
    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=FakeArrayModel
    )

    result = backend.predict({"frac_Ni": 0.2})

    assert result.values == {"hardness": 720.0}
    assert result.uncertainties == {"hardness": 12.5}
    assert backend.targets == ["hardness"]


def test_committee_backend_passes_features_as_single_row():
    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=FakeArrayModel
    )

    backend.predict({"frac_Ni": 0.2, "frac_Al": 0.1})

    assert backend._model.calls == [[{"frac_Ni": 0.2, "frac_Al": 0.1}]]


def test_committee_backend_supports_multiple_targets():
    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=FakeTwoTargetModel
    )

    result = backend.predict({"frac_Ni": 0.2})

    assert result.values == {"hardness": 700.0, "density": 7.9}
    assert result.uncertainties == {"hardness": 10.0, "density": 0.2}


def test_committee_backend_propagates_non_finite_values():
    class NonFiniteModel(FakeArrayModel):
        def predict_array(self, input_data):
            return np.array([[np.nan]]), np.array([[np.inf]]), None, None

    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=NonFiniteModel
    )

    result = backend.predict({"frac_Ni": 0.2})

    assert np.isnan(result.values["hardness"])
    assert np.isinf(result.uncertainties["hardness"])


def test_committee_backend_close_forwards_and_satisfies_protocol():
    backend = CommitteePackageBackend(
        "/tmp/fake.zip", inference_model_cls=FakeArrayModel
    )

    backend.close()

    assert backend._model.closed is True
    assert isinstance(backend, PredictionBackend)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backends.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'env.backends'`

- [ ] **Step 3: Write `env/backends.py`**

```python
"""Typed prediction-backend contract shared by all optimization problems.

A backend turns one candidate (a mapping of feature name to value) into a
:class:`PredictionResult`. Adapters exist so physics simulators, committee
model packages, and future ANN surrogates all satisfy the same protocol.
Uncertainties are diagnostic only — objectives must never let them into
reward computation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PredictionResult:
    """Per-target predictions and diagnostic uncertainties for one candidate."""

    values: dict[str, float]
    uncertainties: dict[str, float]


@runtime_checkable
class PredictionBackend(Protocol):
    def predict(self, features: Mapping[str, float]) -> PredictionResult: ...

    def close(self) -> None: ...


class CommitteePackageBackend:
    """Adapter exposing a committee model-package ZIP as a PredictionBackend.

    Uses ``predict_array`` plus the package's declared ``targets`` so no
    DataFrame column-name convention leaks into consumers.
    """

    def __init__(self, model_package_path, *, inference_model_cls=None):
        if inference_model_cls is None:
            from env import InferenceModel

            inference_model_cls = InferenceModel
        self._model = inference_model_cls(model_package_path)
        self.targets = list(self._model.targets)

    def predict(self, features: Mapping[str, float]) -> PredictionResult:
        y_pred, y_std, *_ = self._model.predict_array([dict(features)])
        values = {
            target: float(y_pred[0][index])
            for index, target in enumerate(self.targets)
        }
        uncertainties = {
            target: float(y_std[0][index])
            for index, target in enumerate(self.targets)
        }
        return PredictionResult(values=values, uncertainties=uncertainties)

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        if callable(close):
            close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backends.py -v`
Expected: 6 PASS

- [ ] **Step 5: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `108 passed`; ruff clean; mypy exit 0.

```bash
git add env/backends.py tests/test_backends.py
git commit -m "feat: add typed PredictionBackend protocol and committee adapter" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Hardness env backend path via value-reader indirection (TDD)

**Files:**
- Modify: `env/material_hardness_env.py`
- Create: `tests/test_material_hardness_env_backend.py`
- Test (must stay green UNMODIFIED): `tests/test_material_hardness_env.py`

**Interfaces:**
- Consumes: `env.backends.PredictionResult` (Task 1).
- Produces: `MaterialHardnessEnv` accepts new optional env-config keys `prediction_backend_cls` (class called with `model_package_path`, must satisfy `PredictionBackend`) and `target_name` (str, default `"hardness"`). Legacy `inference_model_cls` path behaves byte-identically to today.

Design: `step()` keeps its exact current structure. The two prediction touchpoints become indirections:
- `self._raw_predict(composition)` — legacy: `self.model.predict([composition], include_input=False)`; backend: `self.model.predict(composition)` returning `PredictionResult`.
- `self._prediction_value(prediction, column_name, default=_MISSING)` — DataFrame input: existing `_read_prediction_value` logic unchanged; `PredictionResult` input: parse `column_name` as `"<Kind> <target>"`, look up `values` (Kind=`Predicted`) or `uncertainties` (Kind=`Uncertainty`) with casefolded key match, honoring `default` and raising `KeyError(f"Prediction output missing column: {column_name}")` when absent — same message as the DataFrame path.

`step()`'s hardcoded `"Predicted hardness"` / `"Uncertainty hardness"` literals become `f"Predicted {self.target_name}"` / `f"Uncertainty {self.target_name}"` (identical strings for the default `target_name="hardness"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_material_hardness_env_backend.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.backends import PredictionResult
from env.material_hardness_env import MaterialHardnessEnv


class FakeBackend:
    predicted = 720.0
    uncertainty = 12.5

    def __init__(self, model_package_path):
        self.model_package_path = model_package_path
        self.features_seen = []
        self.closed = False

    def predict(self, features):
        self.features_seen.append(dict(features))
        return PredictionResult(
            values={"hardness": self.predicted},
            uncertainties={"hardness": self.uncertainty},
        )

    def close(self):
        self.closed = True


class LegacyFakeModel:
    def __init__(self, model_package_path):
        self.model_package_path = model_package_path

    def predict(self, input_data, include_input=True):
        return pd.DataFrame(
            {
                "Predicted hardness": [FakeBackend.predicted],
                "Uncertainty hardness": [FakeBackend.uncertainty],
            }
        )


BASE_CONFIG = {
    "model_package_path": "/tmp/fake.zip",
    "hardness_threshold": 650.0,
    "reward_scale": 100.0,
    "reward_min": -3.0,
    "reward_max": 3.0,
}


def make_backend_env(backend_cls=FakeBackend, **overrides):
    config = dict(BASE_CONFIG)
    config["prediction_backend_cls"] = backend_cls
    config.update(overrides)
    return MaterialHardnessEnv(config)


def test_backend_and_legacy_paths_produce_identical_step_results():
    backend_env = make_backend_env()
    legacy_env = MaterialHardnessEnv(
        dict(BASE_CONFIG, inference_model_cls=LegacyFakeModel)
    )
    action = np.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.25], dtype=np.float32)

    backend_env.reset(seed=7)
    legacy_env.reset(seed=7)
    b_obs, b_reward, b_term, b_trunc, b_info = backend_env.step(action)
    l_obs, l_reward, l_term, l_trunc, l_info = legacy_env.step(action)

    assert b_reward == l_reward
    assert (b_term, b_trunc) == (l_term, l_trunc)
    assert b_info["predicted_hardness"] == l_info["predicted_hardness"]
    assert b_info["uncertainty_hardness"] == l_info["uncertainty_hardness"]
    assert b_info["reward_unclipped"] == l_info["reward_unclipped"]
    assert b_info["is_success"] == l_info["is_success"]
    assert b_info["composition"] == l_info["composition"]


def test_backend_receives_full_composition_including_fixed_fractions():
    env = make_backend_env()
    env.reset(seed=7)

    env.step(np.zeros(6, dtype=np.float32))

    features = env.model.features_seen[0]
    assert features["frac_Cu"] == 0.0
    assert features["frac_Mo"] == 0.0
    assert set(features) == {
        "frac_Al",
        "frac_Cr",
        "frac_Mn",
        "frac_Fe",
        "frac_Co",
        "frac_Ni",
        "frac_Cu",
        "frac_Mo",
    }


def test_backend_non_finite_prediction_follows_failure_path():
    class NanBackend(FakeBackend):
        predicted = float("nan")

    env = make_backend_env(NanBackend)
    env.reset(seed=7)

    _, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))

    assert reward == -3.0
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is False
    assert info["predicted_hardness"] == 350.0
    assert info["uncertainty_hardness"] == 12.5
    assert "non-finite" in info["error"]


def test_backend_missing_target_raises_missing_column_keyerror():
    class WrongTargetBackend(FakeBackend):
        def predict(self, features):
            return PredictionResult(values={"density": 7.9}, uncertainties={})

    env = make_backend_env(WrongTargetBackend)
    env.reset(seed=7)

    with pytest.raises(
        KeyError, match="Prediction output missing column: Predicted hardness"
    ):
        env.step(np.zeros(6, dtype=np.float32))


def test_backend_close_forwards():
    env = make_backend_env()

    env.close()

    assert env.model.closed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_material_hardness_env_backend.py -v`
Expected: FAIL — `MaterialHardnessEnv` ignores `prediction_backend_cls`, so `__init__` calls `FakeBackend(...)`? No: with no `inference_model_cls`, `__init__` imports the real `InferenceModel` and raises `FileNotFoundError: Model package not found: /tmp/fake.zip`. That collection-free failure IS the expected RED.

- [ ] **Step 3: Modify `env/material_hardness_env.py`**

Add the import (after `from utils.composition_projection import project_bounded_simplex`):

```python
from env.backends import PredictionResult
```

Replace the model-construction block in `__init__` (currently lines 48–57, the `self.model_package_path = ...` through `self.model = inference_model_cls(self.model_package_path)`) with:

```python
        self.model_package_path = config.get(
            "model_package_path",
            os.getenv(MODEL_PACKAGE_PATH_ENV, DEFAULT_MODEL_PACKAGE_PATH),
        )
        self.target_name = str(config.get("target_name", "hardness"))
        prediction_backend_cls = config.get("prediction_backend_cls")
        if prediction_backend_cls is not None:
            self.model = prediction_backend_cls(self.model_package_path)
            self._uses_backend = True
        else:
            inference_model_cls = config.get("inference_model_cls")
            if inference_model_cls is None:
                from env import InferenceModel

                inference_model_cls = InferenceModel
            self.model = inference_model_cls(self.model_package_path)
            self._uses_backend = False
```

In `step()`, replace the prediction call and the three `self._read_prediction_value(...)` call sites:

- `prediction = self.model.predict([composition], include_input=False)` becomes `prediction = self._raw_predict(composition)`
- `self._read_prediction_value(prediction, "Predicted hardness")` becomes `self._prediction_value(prediction, f"Predicted {self.target_name}")`
- `self._read_prediction_value(prediction, "Uncertainty hardness", default=0.0)` becomes `self._prediction_value(prediction, f"Uncertainty {self.target_name}", default=0.0)`
- `self._read_prediction_value(prediction, "Uncertainty hardness")` becomes `self._prediction_value(prediction, f"Uncertainty {self.target_name}")`

Nothing else in `step()` changes.

Add the two indirection methods (place them directly above the existing `_read_prediction_value` staticmethod; keep `_read_prediction_value` itself unchanged):

```python
    def _raw_predict(self, composition: dict[str, float]):
        if self._uses_backend:
            return self.model.predict(composition)
        return self.model.predict([composition], include_input=False)

    def _prediction_value(
        self,
        prediction: object,
        column_name: str,
        default: object = _MISSING,
    ) -> float:
        if isinstance(prediction, PredictionResult):
            kind, _, target = column_name.partition(" ")
            field = (
                prediction.values
                if kind.casefold() == "predicted"
                else prediction.uncertainties
            )
            expected = target.casefold()
            for key, value in field.items():
                if str(key).casefold() == expected:
                    return float(value)
            if default is not _MISSING:
                return float(default)
            raise KeyError(f"Prediction output missing column: {column_name}")
        return self._read_prediction_value(prediction, column_name, default)
```

- [ ] **Step 4: Run the new tests and the untouched legacy tests**

Run: `uv run pytest tests/test_material_hardness_env_backend.py tests/test_material_hardness_env.py -v`
Expected: 5 new PASS + all 13 existing PASS, zero modifications to `tests/test_material_hardness_env.py`.

- [ ] **Step 5: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `113 passed`; ruff clean; mypy exit 0.

```bash
git add env/material_hardness_env.py tests/test_material_hardness_env_backend.py
git commit -m "feat: accept injectable prediction backend in hardness env" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: EEHEMT simulator injection seam (TDD)

**Files:**
- Modify: `env/eehemt_env.py:107-114` (simulator construction only)
- Create: `tests/test_eehemt_simulator_seam.py`
- Test (must stay green UNMODIFIED): `tests/test_env_measure_vds.py`

**Interfaces:**
- Produces: `EEHEMTEnv_Measure_VDS` accepts optional env-config key `simulator_factory` — a callable receiving exactly `(va_file_path, temperature, rs_ext, rd_ext, ir_drop_n_iter, ir_drop_maxfev)` as keyword arguments and returning a simulator object. Default `None` preserves today's `EEHEMTSimulator.from_va_file(...)` construction byte-for-byte.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eehemt_simulator_seam.py`:

```python
from __future__ import annotations

import os

import numpy as np

import env.eehemt_env as eehemt_env_module
from env.eehemt_env import EEHEMTEnv_Measure_VDS
from env.parameter_flow import MeasuredCurveDataset


class FakeSimulator:
    def __init__(self, dataset, defaults):
        self._dataset = dataset
        self._defaults = defaults
        self.last_solver_diagnostics = []

    def modelcard_defaults(self):
        return dict(self._defaults)

    def simulate_current_matrix(self, *, params, vgs, vds_values, current_step):
        self.last_solver_diagnostics = [
            {"converged": True} for _ in range(len(vds_values))
        ]
        return np.zeros_like(self._dataset.current_matrix)


def test_simulator_factory_injects_fake_and_env_still_resets():
    csv_path = os.path.join(os.getcwd(), os.getenv("CSV_FILE_PATH", ""))
    dataset = MeasuredCurveDataset.from_csv(csv_path)
    defaults = {
        name: 0.0 for name in eehemt_env_module.PARAMETER_SPECS.names
    }
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeSimulator(dataset, defaults)

    env = EEHEMTEnv_Measure_VDS(
        {
            "va_file_path": "/nonexistent/never-compiled.va",
            "csv_file_path": csv_path,
            "random_init": False,
            "reduce_obs_err_dim": False,
            "reward_norm": False,
            "simulator_factory": factory,
        }
    )

    observation, info = env.reset(seed=123)

    assert isinstance(env.simulator, FakeSimulator)
    assert captured["va_file_path"] == "/nonexistent/never-compiled.va"
    assert set(captured) == {
        "va_file_path",
        "temperature",
        "rs_ext",
        "rd_ext",
        "ir_drop_n_iter",
        "ir_drop_maxfev",
    }
    assert env.observation_space.contains(observation)
    assert "nrmse" in info
```

(The fake never touches verilogae: a real `from_va_file` on the nonexistent `.va` path would crash, so a passing test proves the factory replaced construction.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eehemt_simulator_seam.py -v`
Expected: FAIL — config key ignored, env calls `EEHEMTSimulator.from_va_file("/nonexistent/never-compiled.va", ...)`, which errors (file not found / compile failure).

- [ ] **Step 3: Modify `env/eehemt_env.py`**

Replace the construction (currently lines 107–114):

```python
        self.simulator = EEHEMTSimulator.from_va_file(
            self.va_file_path,
            temperature=TEMPERATURE,
            rs_ext=self.Rs_ext,
            rd_ext=self.Rd_ext,
            ir_drop_n_iter=self.ir_drop_n_iter,
            ir_drop_maxfev=self.ir_drop_maxfev,
        )
```

with:

```python
        simulator_factory = config.get("simulator_factory")
        if simulator_factory is None:

            def simulator_factory(**kwargs):
                return EEHEMTSimulator.from_va_file(
                    kwargs.pop("va_file_path"), **kwargs
                )

        self.simulator = simulator_factory(
            va_file_path=self.va_file_path,
            temperature=TEMPERATURE,
            rs_ext=self.Rs_ext,
            rd_ext=self.Rd_ext,
            ir_drop_n_iter=self.ir_drop_n_iter,
            ir_drop_maxfev=self.ir_drop_maxfev,
        )
```

- [ ] **Step 4: Run the new test and the untouched EEHEMT suite**

Run: `uv run pytest tests/test_eehemt_simulator_seam.py tests/test_env_measure_vds.py -v`
Expected: 1 new PASS + all 11 existing PASS unmodified.

- [ ] **Step 5: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `114 passed`; ruff clean; mypy exit 0.

```bash
git add env/eehemt_env.py tests/test_eehemt_simulator_seam.py
git commit -m "feat: add simulator_factory injection seam to EEHEMT env" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `env/objectives.py` — the two objective strategies (TDD)

**Files:**
- Create: `env/objectives.py`
- Create: `tests/test_objectives.py`

**Interfaces:**
- Produces (used by Tasks 5–7 and Plan 3):
  - `ThresholdOutcome(reward: float, reward_unclipped: float, success: bool)` — NamedTuple.
  - `ThresholdMaximizeObjective(threshold, scale, reward_min, reward_max)`; class attrs `RANKED_METRIC = "env_runners/max_predicted_hardness"`, `RANKED_ORDER = "max"`; method `evaluate(value: float) -> ThresholdOutcome`; ctor validation messages verbatim (`reward_scale must be positive`, `reward_min must be less than or equal to reward_max`).
  - `NRMSEMinimizeObjective(threshold, reward_min, reward_max, epsilon)`; class attrs `RANKED_METRIC = "env_runners/min_nrmse"`, `RANKED_ORDER = "min"`; methods `reward_from_nrmse(nrmse: float) -> float` and `is_success(nrmse: float) -> bool` (strict `<`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_objectives.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from env.objectives import NRMSEMinimizeObjective, ThresholdMaximizeObjective


def make_threshold_objective(**overrides):
    kwargs = {
        "threshold": 650.0,
        "scale": 100.0,
        "reward_min": -3.0,
        "reward_max": 3.0,
    }
    kwargs.update(overrides)
    return ThresholdMaximizeObjective(**kwargs)


@pytest.mark.parametrize(
    ("value", "expected_reward", "expected_unclipped", "expected_success"),
    [
        (550.0, -1.0, -1.0, False),
        (650.0, 0.0, 0.0, True),
        (1000.0, 3.0, 3.5, True),
        (250.0, -3.0, -4.0, False),
    ],
)
def test_threshold_objective_reproduces_hardness_reward_table(
    value, expected_reward, expected_unclipped, expected_success
):
    outcome = make_threshold_objective().evaluate(value)

    assert outcome.reward == expected_reward
    assert outcome.reward_unclipped == expected_unclipped
    assert outcome.success is expected_success


def test_threshold_objective_success_is_inclusive_at_threshold():
    outcome = make_threshold_objective().evaluate(650.0)

    assert outcome.success is True


def test_threshold_objective_validates_scale_and_bounds():
    with pytest.raises(ValueError, match="reward_scale must be positive"):
        make_threshold_objective(scale=0.0)

    with pytest.raises(
        ValueError, match="reward_min must be less than or equal to reward_max"
    ):
        make_threshold_objective(reward_min=4.0, reward_max=3.0)


def test_threshold_objective_ranking_identity():
    assert (
        ThresholdMaximizeObjective.RANKED_METRIC
        == "env_runners/max_predicted_hardness"
    )
    assert ThresholdMaximizeObjective.RANKED_ORDER == "max"


def make_nrmse_objective(**overrides):
    kwargs = {
        "threshold": 10.0,
        "reward_min": -5.0,
        "reward_max": 5.0,
        "epsilon": 1e-15,
    }
    kwargs.update(overrides)
    return NRMSEMinimizeObjective(**kwargs)


@pytest.mark.parametrize("nrmse", [5.0, 0.05, 37.5])
def test_nrmse_objective_matches_reference_formula(nrmse):
    objective = make_nrmse_objective()

    reward = objective.reward_from_nrmse(nrmse)

    assert reward == float(
        np.clip(-np.log10((nrmse / 100.0) + 1e-15), -5.0, 5.0)
    )


def test_nrmse_objective_clips_at_both_bounds():
    objective = make_nrmse_objective()

    assert objective.reward_from_nrmse(0.0) == 5.0
    assert objective.reward_from_nrmse(1e9) == -5.0


def test_nrmse_objective_success_is_strictly_below_threshold():
    objective = make_nrmse_objective()

    assert objective.is_success(9.999) is True
    assert objective.is_success(10.0) is False


def test_nrmse_objective_ranking_identity():
    assert NRMSEMinimizeObjective.RANKED_METRIC == "env_runners/min_nrmse"
    assert NRMSEMinimizeObjective.RANKED_ORDER == "min"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_objectives.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'env.objectives'`

- [ ] **Step 3: Write `env/objectives.py`**

```python
"""Objective strategies: reward math, success comparison, and ranking identity.

An objective owns exactly three things: how a scalar quality value maps to a
reward, what counts as success, and which RLlib metric ranks checkpoints for
it. Episode control (termination, truncation, solver handling, episode-best
tracking) stays in the environments — see ADR 0003.
"""

from __future__ import annotations

from typing import ClassVar, Literal, NamedTuple

import numpy as np


class ThresholdOutcome(NamedTuple):
    reward: float
    reward_unclipped: float
    success: bool


class ThresholdMaximizeObjective:
    """Maximize a predicted property above a threshold (single-step search)."""

    RANKED_METRIC: ClassVar[str] = "env_runners/max_predicted_hardness"
    RANKED_ORDER: ClassVar[Literal["min", "max"]] = "max"

    def __init__(
        self,
        *,
        threshold: float,
        scale: float,
        reward_min: float,
        reward_max: float,
    ) -> None:
        if scale <= 0.0:
            raise ValueError("reward_scale must be positive")
        if reward_min > reward_max:
            raise ValueError("reward_min must be less than or equal to reward_max")
        self.threshold = float(threshold)
        self.scale = float(scale)
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)

    def evaluate(self, value: float) -> ThresholdOutcome:
        reward_unclipped = (float(value) - self.threshold) / self.scale
        reward = float(np.clip(reward_unclipped, self.reward_min, self.reward_max))
        return ThresholdOutcome(
            reward=reward,
            reward_unclipped=reward_unclipped,
            success=bool(float(value) >= self.threshold),
        )


class NRMSEMinimizeObjective:
    """Minimize NRMSE (percent) of a fitted curve; success strictly below threshold."""

    RANKED_METRIC: ClassVar[str] = "env_runners/min_nrmse"
    RANKED_ORDER: ClassVar[Literal["min", "max"]] = "min"

    def __init__(
        self,
        *,
        threshold: float,
        reward_min: float,
        reward_max: float,
        epsilon: float,
    ) -> None:
        self.threshold = float(threshold)
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)
        self.epsilon = float(epsilon)

    def reward_from_nrmse(self, nrmse: float) -> float:
        nrmse_fraction = float(nrmse) / 100.0
        reward = -np.log10(nrmse_fraction + self.epsilon)
        return float(np.clip(reward, self.reward_min, self.reward_max))

    def is_success(self, nrmse: float) -> bool:
        return bool(float(nrmse) < self.threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_objectives.py -v`
Expected: 13 PASS (threshold: 4 parametrized table cases + inclusive-at-threshold + validation + ranking identity = 7; NRMSE: 3 parametrized formula cases + clip-bounds + strict-success + ranking identity = 6).

- [ ] **Step 5: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `127 passed`; ruff clean; mypy exit 0.

```bash
git add env/objectives.py tests/test_objectives.py
git commit -m "feat: add threshold-maximize and NRMSE-minimize objective strategies" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Hardness env delegates to ThresholdMaximizeObjective

**Files:**
- Modify: `env/material_hardness_env.py`
- Test (must stay green UNMODIFIED): `tests/test_material_hardness_env.py`, `tests/test_material_hardness_env_backend.py`

**Interfaces:**
- Consumes: `ThresholdMaximizeObjective` (Task 4).
- Produces: `MaterialHardnessEnv.objective` attribute (instance of `ThresholdMaximizeObjective`). Env attributes `hardness_threshold`, `reward_scale`, `reward_min`, `reward_max` remain (evaluation code reads them later).

- [ ] **Step 1: Modify `__init__`**

Add the import:

```python
from env.objectives import ThresholdMaximizeObjective
```

Replace the two validation lines (currently `if self.reward_scale <= 0.0: raise ...` and `if self.reward_min > self.reward_max: raise ...`) with objective construction at the same position:

```python
        self.objective = ThresholdMaximizeObjective(
            threshold=self.hardness_threshold,
            scale=self.reward_scale,
            reward_min=self.reward_min,
            reward_max=self.reward_max,
        )
```

(The objective's constructor raises the same two `ValueError` messages at the same point in `__init__`, so `test_init_rejects_invalid_reward_config` passes unchanged.)

- [ ] **Step 2: Delegate in `step()`**

Replace the reward computation block (currently):

```python
        reward_unclipped = (
            predicted_hardness - self.hardness_threshold
        ) / self.reward_scale
        is_success = bool(predicted_hardness >= self.hardness_threshold)
```

with:

```python
        outcome = self.objective.evaluate(predicted_hardness)
        reward_unclipped = outcome.reward_unclipped
        is_success = outcome.success
```

and replace `reward = float(np.clip(reward_unclipped, self.reward_min, self.reward_max))` with:

```python
        reward = outcome.reward
```

The non-finite failure path is untouched (it uses `self.reward_min`/`self.reward_scale`/`self.hardness_threshold` directly, which still exist).

- [ ] **Step 3: Run the two hardness test files**

Run: `uv run pytest tests/test_material_hardness_env.py tests/test_material_hardness_env_backend.py -v`
Expected: ALL PASS unmodified — including the reward-table test, which is the numerical-identity gate.

- [ ] **Step 4: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `127 passed`; ruff clean; mypy exit 0.

```bash
git add env/material_hardness_env.py
git commit -m "refactor: delegate hardness reward math to ThresholdMaximizeObjective" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: EEHEMT env delegates to NRMSEMinimizeObjective

**Files:**
- Modify: `env/eehemt_env.py`
- Test (must stay green UNMODIFIED): `tests/test_env_measure_vds.py`, `tests/test_eehemt_simulator_seam.py`

**Interfaces:**
- Consumes: `NRMSEMinimizeObjective` (Task 4).
- Produces: `EEHEMTEnv_Measure_VDS.objective` attribute. `NRMSE_THRESHOLD`, `REWARD_MIN`, `REWARD_MAX` attributes remain.

- [ ] **Step 1: Modify `env/eehemt_env.py`**

Add the import (alongside the existing `from evaluation.metrics import calculate_nrmse`):

```python
from env.objectives import NRMSEMinimizeObjective
```

In `__init__`, directly after the line `self.REWARD_MAX = float(os.getenv("REWARD_MAX", 5.0))`, add:

```python
        self.objective = NRMSEMinimizeObjective(
            threshold=self.NRMSE_THRESHOLD,
            reward_min=self.REWARD_MIN,
            reward_max=self.REWARD_MAX,
            epsilon=EPSILON,
        )
```

Replace the body of `_scaled_reward_from_nrmse` (keep the method — it is the env's named seam):

```python
    def _scaled_reward_from_nrmse(self, nrmse: float) -> float:
        return self.objective.reward_from_nrmse(nrmse)
```

In `step()`, replace:

```python
        terminated_success = (
            solver_converged and current_nrmse < self.NRMSE_THRESHOLD
        )
```

with:

```python
        terminated_success = solver_converged and self.objective.is_success(
            current_nrmse
        )
```

Nothing else changes: solver-failure `reward = self.REWARD_MIN`, truncation, reward normalization, and episode-best tracking all stay env-owned.

- [ ] **Step 2: Run the EEHEMT test files**

Run: `uv run pytest tests/test_env_measure_vds.py tests/test_eehemt_simulator_seam.py -v`
Expected: ALL PASS unmodified — `test_reward_uses_transformed_nrmse_objective` and `test_termination_uses_nrmse_threshold_not_arcsinh_huber_threshold` are the numerical-identity gates.

- [ ] **Step 3: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `127 passed`; ruff clean; mypy exit 0.

```bash
git add env/eehemt_env.py
git commit -m "refactor: delegate EEHEMT reward and success to NRMSEMinimizeObjective" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Single-source checkpoint metrics from objectives + ADR (TDD)

**Files:**
- Modify: `problems/hardness.py`, `problems/eehemt.py`
- Modify: `training/hardness_ppo.py` (`build_checkpoint_config`), `training/eehemt_ppo.py` (`build_checkpoint_config`)
- Create: `docs/adr/0003-problem-registry-backends-objectives.md`
- Modify: `.codebase-memory/adr.md` (append entry following that file's existing format)
- Create: `tests/test_checkpoint_metric_single_source.py`
- Test (must stay green UNMODIFIED): `tests/test_problem_builtin_specs.py`, `tests/test_train_ppo_config.py`

**Interfaces:**
- Consumes: `ThresholdMaximizeObjective.RANKED_METRIC/RANKED_ORDER`, `NRMSEMinimizeObjective.RANKED_METRIC/RANKED_ORDER` (Task 4).
- Produces: the checkpoint metric name exists in exactly one place per objective; specs and Tune checkpoint configs both read it from the objective class.

- [ ] **Step 1: Write the failing test**

Create `tests/test_checkpoint_metric_single_source.py`:

```python
import problems  # noqa: F401
from env.objectives import NRMSEMinimizeObjective, ThresholdMaximizeObjective
from problems import registry
from training import eehemt_ppo, hardness_ppo


def test_hardness_metric_single_sourced_from_objective():
    spec = registry.get("hardness")
    checkpoint_config = hardness_ppo.build_checkpoint_config()

    assert spec.checkpoint_metric == ThresholdMaximizeObjective.RANKED_METRIC
    assert spec.checkpoint_order == ThresholdMaximizeObjective.RANKED_ORDER
    assert (
        checkpoint_config.checkpoint_score_attribute
        == ThresholdMaximizeObjective.RANKED_METRIC
    )
    assert (
        checkpoint_config.checkpoint_score_order
        == ThresholdMaximizeObjective.RANKED_ORDER
    )


def test_eehemt_metric_single_sourced_from_objective():
    spec = registry.get("eehemt")
    checkpoint_config = eehemt_ppo.build_checkpoint_config()

    assert spec.checkpoint_metric == NRMSEMinimizeObjective.RANKED_METRIC
    assert spec.checkpoint_order == NRMSEMinimizeObjective.RANKED_ORDER
    assert (
        checkpoint_config.checkpoint_score_attribute
        == NRMSEMinimizeObjective.RANKED_METRIC
    )
    assert (
        checkpoint_config.checkpoint_score_order
        == NRMSEMinimizeObjective.RANKED_ORDER
    )
```

Run: `uv run pytest tests/test_checkpoint_metric_single_source.py -v`
Expected: PASS already for equality of strings — BUT this test is a characterization of the END state; before the edits it passes because the literals happen to match. Proceed to the edits anyway: the point is that after Step 2 the literals no longer exist outside `env/objectives.py`. (Grep gate in Step 3 enforces that.)

- [ ] **Step 2: Make specs and checkpoint configs read the objective attrs**

`problems/hardness.py` — add `from env.objectives import ThresholdMaximizeObjective` and change the two fields:

```python
        checkpoint_metric=ThresholdMaximizeObjective.RANKED_METRIC,
        checkpoint_order=ThresholdMaximizeObjective.RANKED_ORDER,
```

`problems/eehemt.py` — add `from env.objectives import NRMSEMinimizeObjective` and change:

```python
        checkpoint_metric=NRMSEMinimizeObjective.RANKED_METRIC,
        checkpoint_order=NRMSEMinimizeObjective.RANKED_ORDER,
```

`training/hardness_ppo.py` — add `from env.objectives import ThresholdMaximizeObjective` at module top (safe: objectives imports nothing from training) and change `build_checkpoint_config`:

```python
def build_checkpoint_config() -> tune.CheckpointConfig:
    return tune.CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute=ThresholdMaximizeObjective.RANKED_METRIC,
        checkpoint_score_order=ThresholdMaximizeObjective.RANKED_ORDER,
    )
```

`training/eehemt_ppo.py` — same with `NRMSEMinimizeObjective`.

- [ ] **Step 3: Grep gate — the metric literals live only in objectives + tests**

Run:

```bash
grep -rn --include="*.py" --exclude-dir=.venv --exclude-dir=.git \
  -e "env_runners/max_predicted_hardness" -e "env_runners/min_nrmse" . \
  | grep -v "^\./tests/" | grep -v "^\./env/objectives\.py"
```

Expected: no output (the two metric literals survive only in `env/objectives.py` and in test files).

- [ ] **Step 4: Write the ADR**

Create `docs/adr/0003-problem-registry-backends-objectives.md`:

```markdown
# 0003 — Problem registry, prediction backends, and objective strategies

Date: 2026-07-23
Status: accepted

## Context

The PPO harness supported two problems (`eehemt`, `hardness`) through
hardcoded `--env` dispatch, copy-pasted training assembly, reward/termination
logic inlined in each env's `step()`, and checkpoint-metric names repeated as
string literals across training configs, callbacks, and evaluation. Adding a
third problem (new material, objective, or prediction model) required editing
shared harness code in at least four places and copying an entire vertical
slice. Two independent architecture reviews (deep-reasoner, Codex) converged
on the same remedy.

## Decision

1. **Problem registry** (`problems/`): `--env <name>` resolves via
   `problems.registry` (`ProblemSpec` — frozen dataclass carrying the training
   module, W&B project, checkpoint metric/order, and the four assembly
   callables). Built-ins self-register on package import;
   `select_training_module` stays as a facade.
2. **Generic PPO assembly**: `training/ppo_common.build_base_ppo_config` owns
   the shared PPOConfig chain; problem modules delegate, injecting env class,
   env config, callback, and evaluation function.
3. **Prediction backends** (`env/backends.py`): a `PredictionBackend` protocol
   (`predict(features) -> PredictionResult`, `close()`) with
   `CommitteePackageBackend` wrapping `InferenceModel` via `predict_array` +
   declared `targets`. `MaterialHardnessEnv` accepts `prediction_backend_cls`
   alongside the legacy `inference_model_cls` path; both flow through one
   `step()` body via a value-reader indirection. `EEHEMTEnv_Measure_VDS`
   accepts `simulator_factory`, closing the injection asymmetry and giving
   EEHEMT its first test seam. Uncertainty remains diagnostic-only.
4. **Objective strategies** (`env/objectives.py`):
   `ThresholdMaximizeObjective` and `NRMSEMinimizeObjective` own reward math,
   success comparison, and the ranked-metric identity
   (`RANKED_METRIC`/`RANKED_ORDER`). Checkpoint configs and problem specs read
   the metric from the objective class — the string exists in one place.
   **Refinement of the original design spec:** episode control (termination,
   truncation, solver-failure penalty, episode-best tracking) deliberately
   stays in the envs. Forcing the episodic, solver-coupled EEHEMT problem and
   the single-step hardness search into one generic env/objective contract was
   judged an over-generalization risk.

## Consequences

- A new problem needs: a problem package registering a `ProblemSpec`, an env
  (or reuse), a backend adapter satisfying `PredictionBackend`, and an
  objective (reusing a built-in where semantics match). No shared-file edits.
- Guardrails preserved and test-locked: EEHEMT NRMSE reward/termination
  semantics (ADR 0001), hardness six-fraction action space, NoFilter default,
  float32 finite-bound observations, verbatim checkpoint-metric strings.
- `success_rate_650` is still emitted for dashboard continuity even though the
  threshold is configurable (600 in the current `.env`); a threshold-agnostic
  `success_rate` metric is planned alongside the evaluation/callback
  parameterization (Plan 3).
```

- [ ] **Step 5: Append to `.codebase-memory/adr.md`**

Read the file first; append an entry mirroring its existing per-ADR format, titled `0003 — Problem registry, prediction backends, and objective strategies`, summarizing the four decisions and the episode-control refinement in the same style/length as the existing entries.

- [ ] **Step 6: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `129 passed`; ruff clean; mypy exit 0.

```bash
git add problems/hardness.py problems/eehemt.py training/hardness_ppo.py training/eehemt_ppo.py tests/test_checkpoint_metric_single_source.py docs/adr/0003-problem-registry-backends-objectives.md .codebase-memory/adr.md
git commit -m "refactor: single-source checkpoint metrics from objectives and add ADR 0003" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Plan wrap-up verification

**Files:**
- Modify: `docs/superpowers/plans/2026-07-23-p2-backends-and-objectives.md` (tick checkboxes)

- [ ] **Step 1: Full verification trio from a clean state**

```bash
uv run pytest && uv run ruff check . && uv run mypy .
```

Expected: `129 passed`, ruff clean, mypy exit 0.

- [ ] **Step 2: Confirm forbidden files untouched**

```bash
git log --stat c2ae3ca..HEAD -- utils/callbacks.py utils/hardness_callbacks.py evaluation/
```

Expected: empty output. `git status --short` still shows `.claude/` and `CLAUDE.md` as `??` only.

- [ ] **Step 3: Tick all checkboxes in this plan and commit**

```bash
git add docs/superpowers/plans/2026-07-23-p2-backends-and-objectives.md
git commit -m "docs: mark P2 backends/objectives plan complete" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
