import dataclasses
import types

import pytest

from problems import registry


def _dummy_spec(name="dummy"):
    module = types.ModuleType(f"{name}_module")
    return registry.ProblemSpec(
        name=name,
        module=module,
        wandb_project=f"{name}-project",
        checkpoint_metric=f"env_runners/{name}_metric",
        checkpoint_order="max",
        add_env_args=lambda parser, current_dir: parser,
        build_env_config=lambda args: {},
        build_ppo_config=lambda args, **kwargs: None,
        build_checkpoint_config=lambda: None,
    )


@pytest.fixture(autouse=True)
def _isolated_registry():
    saved = registry.snapshot()
    registry.clear()
    yield
    registry.restore(saved)


def test_register_and_get_roundtrip():
    spec = _dummy_spec()
    registry.register(spec)

    assert registry.get("dummy") is spec


def test_register_rejects_duplicate_name():
    registry.register(_dummy_spec())

    with pytest.raises(ValueError, match="Problem already registered: dummy"):
        registry.register(_dummy_spec())


def test_get_unknown_name_uses_legacy_error_message():
    with pytest.raises(
        ValueError, match="Unsupported training environment: nosuch"
    ):
        registry.get("nosuch")


def test_names_are_sorted():
    registry.register(_dummy_spec("zeta"))
    registry.register(_dummy_spec("alpha"))

    assert registry.names() == ["alpha", "zeta"]


def test_spec_is_immutable():
    spec = _dummy_spec()

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"
