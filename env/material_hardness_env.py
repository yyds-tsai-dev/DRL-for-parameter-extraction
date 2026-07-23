from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from env.backends import PredictionResult
from utils.composition_projection import project_bounded_simplex

TUNABLE_FRACTION_NAMES = (
    "frac_Al",
    "frac_Cr",
    "frac_Mn",
    "frac_Fe",
    "frac_Co",
    "frac_Ni",
)
FIXED_FRACTIONS = {"frac_Cu": 0.0, "frac_Mo": 0.0}
DEFAULT_MODEL_PACKAGE_PATH = "env/hardness/XGB_model_selection_package.zip"
MODEL_PACKAGE_PATH_ENV = "HARDNESS_MODEL_PACKAGE_PATH"
_MISSING = object()


class MaterialHardnessEnv(gym.Env):
    """Single-step environment for hardness-oriented material composition search."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()

        config = config or {}
        self.lower = float(config.get("lower", 0.05))
        self.upper = float(config.get("upper", 0.35))
        self.target_sum = float(config.get("target_sum", 1.0))
        self.hardness_threshold = float(config.get("hardness_threshold", 650.0))
        self.reward_scale = float(config.get("reward_scale", 100.0))
        self.reward_min = float(config.get("reward_min", -3.0))
        self.reward_max = float(config.get("reward_max", 3.0))
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive")
        if self.reward_min > self.reward_max:
            raise ValueError("reward_min must be less than or equal to reward_max")

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

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(len(TUNABLE_FRACTION_NAMES),),
            dtype=np.float32,
        )
        self.observation_space = Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        composition = self._composition_from_action(action)
        prediction = self._raw_predict(composition)

        predicted_hardness = self._prediction_value(
            prediction, f"Predicted {self.target_name}"
        )
        if not np.isfinite(predicted_hardness):
            raw_predicted_hardness = predicted_hardness
            failure_predicted_hardness = (
                self.hardness_threshold + self.reward_min * self.reward_scale
            )
            uncertainty_hardness = self._prediction_value(
                prediction,
                f"Uncertainty {self.target_name}",
                default=0.0,
            )
            if not np.isfinite(uncertainty_hardness):
                uncertainty_hardness = 0.0
            info = {
                "composition": composition,
                "predicted_hardness": failure_predicted_hardness,
                "raw_predicted_hardness": str(raw_predicted_hardness),
                "uncertainty_hardness": uncertainty_hardness,
                "reward_unclipped": float(self.reward_min),
                "is_success": False,
                "error": "non-finite predicted hardness",
            }
            return self._observation(), self.reward_min, True, False, info

        uncertainty_hardness = self._prediction_value(
            prediction, f"Uncertainty {self.target_name}"
        )
        reward_unclipped = (
            predicted_hardness - self.hardness_threshold
        ) / self.reward_scale
        is_success = bool(predicted_hardness >= self.hardness_threshold)
        info = {
            "composition": composition,
            "predicted_hardness": predicted_hardness,
            "uncertainty_hardness": uncertainty_hardness,
            "reward_unclipped": reward_unclipped,
            "is_success": is_success,
        }

        reward = float(np.clip(reward_unclipped, self.reward_min, self.reward_max))
        return self._observation(), reward, True, False, info

    def _composition_from_action(self, action) -> dict[str, float]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (len(TUNABLE_FRACTION_NAMES),):
            raise ValueError(
                f"action shape must be ({len(TUNABLE_FRACTION_NAMES)},)"
            )
        tunable_fractions = project_bounded_simplex(
            action,
            lower=self.lower,
            upper=self.upper,
            target_sum=self.target_sum,
        )
        composition = {
            name: float(value)
            for name, value in zip(
                TUNABLE_FRACTION_NAMES,
                tunable_fractions,
                strict=True,
            )
        }
        composition.update(FIXED_FRACTIONS)
        return composition

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
        super().close()

    def _observation(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

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

    @staticmethod
    def _read_prediction_value(
        prediction: pd.DataFrame,
        column_name: str,
        default: object = _MISSING,
    ) -> float:
        if not isinstance(prediction, pd.DataFrame):
            prediction = pd.DataFrame(prediction)

        expected = column_name.casefold()
        for column in prediction.columns:
            if str(column).casefold() == expected:
                return float(prediction.iloc[0][column])
        if default is not _MISSING:
            return float(default)
        raise KeyError(f"Prediction output missing column: {column_name}")
