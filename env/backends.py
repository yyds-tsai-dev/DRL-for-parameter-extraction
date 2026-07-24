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
    """Per-target predictions and diagnostic uncertainties for one candidate.

    Frozen at the dataclass level; the inner dicts are not deep-frozen.
    """

    values: dict[str, float]
    uncertainties: dict[str, float]


@runtime_checkable
class PredictionBackend(Protocol):
    """Structural contract; runtime isinstance checks verify method presence only."""

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
