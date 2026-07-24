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
