"""Problem registry: maps ``--env`` names to their training assembly parts.

A problem registers a :class:`ProblemSpec` once at import time (see
``problems/__init__.py``). ``train_ppo.py`` and ``training/ppo_common.py``
resolve everything problem-specific through this registry, so adding a new
problem requires no edits to shared harness code.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    module: ModuleType
    wandb_project: str
    checkpoint_metric: str
    checkpoint_order: Literal["min", "max"]
    add_env_args: Callable
    build_env_config: Callable
    build_ppo_config: Callable
    build_checkpoint_config: Callable


_REGISTRY: dict[str, ProblemSpec] = {}


def register(spec: ProblemSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"Problem already registered: {spec.name}")
    _REGISTRY[spec.name] = spec


def get(name: str) -> ProblemSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unsupported training environment: {name}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def snapshot() -> dict[str, ProblemSpec]:
    """Test helper: capture current registrations."""
    return dict(_REGISTRY)


def clear() -> None:
    """Test helper: drop all registrations."""
    _REGISTRY.clear()


def restore(saved: dict[str, ProblemSpec]) -> None:
    """Test helper: reinstate a snapshot taken with :func:`snapshot`."""
    _REGISTRY.clear()
    _REGISTRY.update(saved)
