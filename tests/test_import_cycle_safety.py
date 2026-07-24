"""Regression guard: importing ppo_common first must not create an import cycle.

training.ppo_common defers its problems-registry import into
build_common_arg_parser (see ADR 0003); if someone moves it back to module
top, the cycle ppo_common -> problems -> training.*_ppo -> ppo_common breaks
any process that imports ppo_common first. A subprocess gives us a clean
interpreter to prove the import order is safe.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_ppo_common_imports_cleanly_before_problems():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import training.ppo_common; import training.hardness_ppo; "
            "import problems; import train_ppo",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )

    assert result.returncode == 0, result.stderr
