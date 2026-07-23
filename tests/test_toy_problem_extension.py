"""Acceptance proof for the pluggable-problem architecture (ADR 0003).

A complete third problem — env, backend, objective, training assembly — is
assembled and registered HERE, in test code only. If these tests pass without
any edit to train_ppo.py, training/, problems/, env/, or evaluation/, the
"adapter + registration, zero shared-file edits" contract holds.
"""

from __future__ import annotations

import types

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box

import problems  # noqa: F401  (registers builtins)
import train_ppo
from env.backends import PredictionResult
from env.objectives import ThresholdMaximizeObjective
from problems import registry
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from training.ppo_common import build_base_ppo_config


class ToyBackend:
    """Deterministic stand-in for an ANN surrogate."""

    def __init__(self, model_package_path):
        self.model_package_path = model_package_path

    def predict(self, features):
        strength = 100.0 * float(sum(features.values()))
        return PredictionResult(
            values={"strength": strength}, uncertainties={"strength": 1.0}
        )

    def close(self):
        pass


class ToyStrengthEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.backend = ToyBackend(config.get("model_package_path", "unused"))
        self.objective = ThresholdMaximizeObjective(
            threshold=float(config.get("strength_threshold", 50.0)),
            scale=10.0,
            reward_min=-1.0,
            reward_max=1.0,
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        features = {"a": float(action[0]), "b": float(action[1])}
        result = self.backend.predict(features)
        outcome = self.objective.evaluate(result.values["strength"])
        info = {"predicted_strength": result.values["strength"]}
        return np.zeros(1, dtype=np.float32), outcome.reward, True, False, info


def _toy_add_env_args(parser, current_dir):
    parser.add_argument("--strength_threshold", type=float, default=50.0)
    return parser


def _toy_build_env_config(args):
    return {"strength_threshold": args.strength_threshold}


def _toy_build_ppo_config(args, *, num_learners, num_gpus_per_learner):
    return build_base_ppo_config(
        args,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        env_cls=ToyStrengthEnv,
        env_config=_toy_build_env_config(args),
        callbacks_class=DefaultCallbacks,
        custom_evaluation_function=None,
    )


def _toy_build_checkpoint_config():
    return types.SimpleNamespace(
        checkpoint_score_attribute=ThresholdMaximizeObjective.RANKED_METRIC,
        checkpoint_score_order=ThresholdMaximizeObjective.RANKED_ORDER,
    )


def _build_toy_module():
    module = types.ModuleType("toy_strength_problem")
    module.TOY_WANDB_PROJECT = "PPO_for_toy_strength"
    module.add_env_args = _toy_add_env_args
    module.build_env_config = _toy_build_env_config
    module.build_ppo_config = _toy_build_ppo_config
    module.build_checkpoint_config = _toy_build_checkpoint_config
    return module


def _toy_spec():
    module = _build_toy_module()
    return registry.ProblemSpec(
        name="toy_strength",
        module=module,
        wandb_project=module.TOY_WANDB_PROJECT,
        checkpoint_metric=ThresholdMaximizeObjective.RANKED_METRIC,
        checkpoint_order=ThresholdMaximizeObjective.RANKED_ORDER,
        add_env_args=module.add_env_args,
        build_env_config=module.build_env_config,
        build_ppo_config=module.build_ppo_config,
        build_checkpoint_config=module.build_checkpoint_config,
    )


@pytest.fixture()
def toy_registered():
    saved = registry.snapshot()
    registry.register(_toy_spec())
    yield
    registry.restore(saved)


def test_toy_problem_reaches_cli_and_dispatch_without_shared_edits(toy_registered):
    parser = train_ppo.build_arg_parser("/project", ["--env", "toy_strength"])
    args = parser.parse_args(["--env", "toy_strength", "--strength_threshold", "75.0"])

    assert "toy_strength" in registry.names()
    assert args.env == "toy_strength"
    assert args.strength_threshold == 75.0
    assert (
        train_ppo.select_training_module("toy_strength").TOY_WANDB_PROJECT
        == "PPO_for_toy_strength"
    )


def test_toy_problem_builds_ppo_config_through_generic_assembly(toy_registered):
    from types import SimpleNamespace

    args = SimpleNamespace(
        num_env_runners=1,
        observation_filter="NoFilter",
        train_batch_size_per_learner=64,
        num_epochs=1,
        minibatch_size=32,
        lr=1e-5,
        entropy_coeff=0.0,
        grad_clip=1.0,
        vf_loss_coeff=0.1,
        evaluation_interval=1,
        evaluation_num_env_runners=1,
        strength_threshold=50.0,
    )

    spec = registry.get("toy_strength")
    config = spec.build_ppo_config(args, num_learners=1, num_gpus_per_learner=0.0)

    assert config.env is ToyStrengthEnv
    assert config.observation_filter == "NoFilter"
    checkpoint = spec.build_checkpoint_config()
    assert (
        checkpoint.checkpoint_score_attribute
        == ThresholdMaximizeObjective.RANKED_METRIC
    )


def test_toy_env_episode_uses_backend_and_objective(toy_registered):
    env = ToyStrengthEnv({"strength_threshold": 50.0})
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(
        np.array([0.4, 0.3], dtype=np.float32)
    )

    assert info["predicted_strength"] == pytest.approx(70.0)
    assert reward == pytest.approx(1.0)  # (70-50)/10 = 2.0 clipped to reward_max=1.0
    assert terminated is True and truncated is False
