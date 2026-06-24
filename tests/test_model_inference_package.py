from __future__ import annotations

import subprocess
import sys
import tomllib
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


def test_tensorflow_is_optional_core_dependency() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject_path = project_root / "pyproject.toml"
    pyproject_text = pyproject_path.read_text()
    config = tomllib.loads(pyproject_text)

    optional_index = pyproject_text.index("[project.optional-dependencies]")
    dependency_groups_index = pyproject_text.index("[dependency-groups]")

    core_dependencies = config["project"]["dependencies"]
    assert all("tensorflow" not in dependency.lower() for dependency in core_dependencies)
    assert optional_index < dependency_groups_index
    assert config["project"]["optional-dependencies"]["keras-inference"] == [
        "tensorflow>=2.19.0",
    ]
