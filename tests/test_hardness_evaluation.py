import json
from datetime import date

import pandas as pd
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS

from evaluation.hardness_evaluation import (
    _episode_final_info,
    evaluate_and_save_hardness,
    save_hardness_evaluation,
)


def test_save_hardness_evaluation_writes_csv_and_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record = {
        "evaluation_index": 3,
        "training_iteration": 14,
        "best_composition": {"frac_Al": 0.2, "frac_Ni": 0.18},
        "predicted_hardness": 720.0,
        "max_predicted_hardness": 720.0,
        "uncertainty_hardness": 11.0,
        "success_rate_650": 1.0,
    }

    paths = save_hardness_evaluation(record, output_dir=tmp_path / "nested")

    assert paths.csv_path == tmp_path / "nested" / "eval_000003_iter_000014.csv"
    assert paths.json_path == tmp_path / "nested" / "eval_000003_iter_000014.json"
    csv_row = pd.read_csv(paths.csv_path).iloc[0].to_dict()
    assert "best_composition" not in csv_row
    assert csv_row["best_frac_Al"] == 0.2
    assert csv_row["best_frac_Ni"] == 0.18
    assert csv_row["predicted_hardness"] == 720.0
    assert json.loads(paths.json_path.read_text()) == record

    default_paths = save_hardness_evaluation({**record, "evaluation_index": 4})

    assert default_paths.csv_path.parent == (
        tmp_path / "result" / "hardness-evaluation" / date.today().isoformat()
    )
    assert default_paths.csv_path.exists()
    assert default_paths.json_path.exists()


def test_save_hardness_evaluation_writes_strict_json_for_non_finite_values(
    tmp_path,
):
    record = {
        "evaluation_index": 1,
        "training_iteration": 2,
        "best_composition": {"frac_Al": float("nan"), "frac_Ni": float("inf")},
        "predicted_hardness": float("nan"),
        "max_predicted_hardness": float("inf"),
        "uncertainty_hardness": float("-inf"),
        "history": [1.0, float("nan"), {"nested": float("inf")}],
        "success_rate_650": 0.0,
    }

    paths = save_hardness_evaluation(record, output_dir=tmp_path)

    json_text = paths.json_path.read_text()
    assert "NaN" not in json_text
    assert "Infinity" not in json_text
    payload = json.loads(json_text)
    assert payload["predicted_hardness"] is None
    assert payload["max_predicted_hardness"] is None
    assert payload["uncertainty_hardness"] is None
    assert payload["best_composition"] == {"frac_Al": None, "frac_Ni": None}
    assert payload["history"] == [1.0, None, {"nested": None}]


def test_save_hardness_evaluation_default_dir_uses_project_root(
    tmp_path,
    monkeypatch,
):
    project_root = tmp_path / "project"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    record = {
        "evaluation_index": 1,
        "training_iteration": 2,
        "best_composition": {},
        "predicted_hardness": 720.0,
    }

    paths = save_hardness_evaluation(record)

    assert paths.csv_path.parent == (
        project_root / "result" / "hardness-evaluation" / date.today().isoformat()
    )
    assert paths.csv_path.exists()
    assert paths.json_path.exists()


class FakeGetInfosEpisode:
    def __init__(self, info, *, env_steps, agent_steps):
        self.info = info
        self._env_steps = env_steps
        self._agent_steps = agent_steps

    def get_infos(self, indices):
        assert indices == -1
        return self.info

    def env_steps(self):
        return self._env_steps

    def agent_steps(self):
        return self._agent_steps


class FakeInfosEpisode:
    def __init__(self, infos):
        self.infos = infos


def test_episode_final_info_falls_back_to_infos():
    final_info = {"predicted_hardness": 710.0}
    episode = FakeInfosEpisode([{"predicted_hardness": 650.0}, final_info])

    assert _episode_final_info(episode) is final_info


class FakeEnvRunner:
    def __init__(self, episodes, metrics):
        self.episodes = episodes
        self.metrics = metrics
        self.sample_calls = []

    def sample(self, *, num_episodes):
        self.sample_calls.append(num_episodes)
        return self.episodes

    def get_metrics(self):
        return self.metrics


class FakeEvalWorkers:
    def __init__(self, runner):
        self.runner = runner
        self.foreach_calls = []

    def foreach_env_runner(self, *args, **kwargs):
        func = kwargs.get("func")
        if func is None:
            func = args[0]
        self.foreach_calls.append(kwargs)
        return [func(self.runner)]


class FakeMetrics:
    def __init__(self):
        self.aggregate_calls = []
        self.peek_calls = []

    def aggregate(self, metrics, *, key):
        self.aggregate_calls.append((metrics, key))

    def peek(self, key):
        self.peek_calls.append(key)
        return {"episode_return_mean": 720.0}


class FakeAlgorithm:
    def __init__(self):
        self.iteration = 7
        self.metrics = FakeMetrics()


def test_evaluate_and_save_hardness_writes_artifact(tmp_path):
    final_info = {
        "composition": {"frac_Al": 0.2, "frac_Ni": 0.18},
        "predicted_hardness": 720.0,
        "max_predicted_hardness": 720.0,
        "uncertainty_hardness": 11.0,
        "is_success": True,
    }
    episode = FakeGetInfosEpisode(final_info, env_steps=5, agent_steps=6)
    runner = FakeEnvRunner([episode], metrics={"runner_metric": 2})
    eval_workers = FakeEvalWorkers(runner)
    algorithm = FakeAlgorithm()

    eval_results, env_steps, agent_steps = evaluate_and_save_hardness(
        algorithm,
        eval_workers,
        output_dir=tmp_path,
    )

    assert eval_results == {"episode_return_mean": 720.0}
    assert env_steps == 5
    assert agent_steps == 6
    assert runner.sample_calls == [1]
    metric_key = (EVALUATION_RESULTS, ENV_RUNNER_RESULTS)
    assert algorithm.metrics.aggregate_calls == [([{"runner_metric": 2}], metric_key)]
    assert algorithm.metrics.peek_calls == [metric_key]
    assert algorithm._hardness_evaluation_index == 1

    json_payload = json.loads((tmp_path / "eval_000001_iter_000008.json").read_text())
    assert json_payload == {
        "evaluation_index": 1,
        "training_iteration": 8,
        "best_composition": {"frac_Al": 0.2, "frac_Ni": 0.18},
        "predicted_hardness": 720.0,
        "max_predicted_hardness": 720.0,
        "uncertainty_hardness": 11.0,
        "success_rate_650": 1.0,
    }
    csv_row = pd.read_csv(tmp_path / "eval_000001_iter_000008.csv").iloc[0].to_dict()
    assert csv_row["best_frac_Al"] == 0.2
    assert csv_row["best_frac_Ni"] == 0.18

    evaluate_and_save_hardness(algorithm, eval_workers, output_dir=tmp_path)

    assert algorithm._hardness_evaluation_index == 2
    assert (tmp_path / "eval_000002_iter_000008.csv").exists()
    assert (tmp_path / "eval_000002_iter_000008.json").exists()
