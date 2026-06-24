from types import SimpleNamespace

import pytest

from training.ppo_common import (
    build_common_arg_parser,
    build_wandb_callback,
    resolve_learner_resources,
)


def _callback_value(callback, name):
    if hasattr(callback, name):
        return getattr(callback, name)
    kwargs = getattr(callback, "kwargs", {})
    if name in kwargs:
        return kwargs[name]
    pytest.fail(f"callback does not expose {name!r}")


def test_common_parser_wandb_api_key_defaults_from_environment(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "from-env")
    parser = build_common_arg_parser("/project")

    args = parser.parse_args([])

    assert args.wandb_api_key == "from-env"


def test_common_parser_env_defaults_to_hardness_and_can_select_eehemt():
    parser = build_common_arg_parser("/project")

    default_args = parser.parse_args([])
    override_args = parser.parse_args(["--env", "eehemt"])

    assert default_args.env == "hardness"
    assert override_args.env == "eehemt"


def test_common_parser_wandb_api_key_cli_override(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "from-env")
    parser = build_common_arg_parser("/project")

    args = parser.parse_args(["--wandb_api_key", "from-cli"])

    assert args.wandb_api_key == "from-cli"


def test_build_wandb_callback_uses_project_name_and_api_key():
    args = SimpleNamespace(wandb_api_key="secret")

    callback = build_wandb_callback(args, project_name="project-name")

    assert _callback_value(callback, "project") == "project-name"
    assert _callback_value(callback, "api_key") == "secret"


def test_resolve_learner_resources_uses_env_learners_without_cuda(monkeypatch):
    monkeypatch.setattr("training.ppo_common.th.cuda.device_count", lambda: 0)
    monkeypatch.setenv("NUM_LEARNERS", "2")

    assert resolve_learner_resources() == (2, 0.0)
