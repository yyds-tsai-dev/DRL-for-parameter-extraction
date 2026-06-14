from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_model_inference_package_exports_public_api() -> None:
    from env import InferenceModel, predict

    assert InferenceModel.__name__ == "InferenceModel"
    assert callable(predict)


def test_model_inference_cli_help_runs_from_repo_root() -> None:
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/run_model_inference.py", "--help"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run local inference with a trained model package." in result.stdout
