import logging

import numpy as np
import pytest
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS

from evaluation.iv_curve_evaluation import (
    _episode_agent_steps,
    _episode_env_steps,
    _episode_final_info,
    _plot_dir,
    evaluate_and_plot_iv_curve,
)


class FakeEpisode:
    def __init__(self, infos=None, env_steps=0, agent_steps=0, callable_steps=True):
        self.infos = infos or []
        if callable_steps:
            self._env_steps = env_steps
            self._agent_steps = agent_steps
        else:
            self.env_steps = env_steps
            self.agent_steps = agent_steps

    def env_steps(self):
        return self._env_steps

    def agent_steps(self):
        return self._agent_steps


class FakeGetInfosEpisode:
    def __init__(self, info):
        self.info = info

    def get_infos(self, indices):
        assert indices == -1
        return self.info


class FakeEnv:
    def __init__(self, *, curve_condition_values=None, vds=None):
        self.curve_condition_values = curve_condition_values
        self.vds = vds
        self.vgs = np.array([0.0, 1.0])
        self.i_meas_dict = {
            0.1: np.array([1.0e-3, 2.0e-3]),
            0.2: np.array([2.0e-3, 3.0e-3]),
        }

    def _get_plot_data_matrix(self):
        plot_data = {
            "vgs": self.vgs,
            "i_meas_dict": self.i_meas_dict,
        }
        if hasattr(self, "last_i_sim_current_matrix"):
            plot_data["i_sim_current_matrix"] = self.last_i_sim_current_matrix
        if hasattr(self, "episode_best_i_sim_current_matrix"):
            plot_data["episode_best_i_sim_current_matrix"] = (
                self.episode_best_i_sim_current_matrix
            )
        return plot_data


class FakeEnvWithoutPlotData:
    curve_condition_values = [0.1, 0.2]


class FakeUnwrapped:
    def __init__(self, target):
        self.unwrapped = target


class FakeVectorEnv:
    def __init__(self, target):
        self.unwrapped = self
        self.envs = [FakeUnwrapped(target)]


class FakeEnvRunner:
    def __init__(self, episodes, metrics, env):
        self.episodes = episodes
        self.metrics = metrics
        self.env = env
        self.sample_calls = []

    def sample(self, *, num_episodes):
        self.sample_calls.append(num_episodes)
        return self.episodes

    def get_metrics(self):
        return self.metrics


class FakeEvalWorkerGroup:
    def __init__(self, *runners, actor_ids=None, healthy_worker_ids=None, local_runner=None):
        self.runners = list(runners)
        self._worker_manager = FakeWorkerManager(
            actor_ids if actor_ids is not None else list(range(1, len(runners) + 1))
        )
        self._healthy_worker_ids = (
            healthy_worker_ids if healthy_worker_ids is not None else self._worker_manager.actor_ids()
        )
        self.local_runner = local_runner
        self.local_env_runner_values = []
        self.remote_worker_ids_values = []

    def foreach_env_runner(self, *, func, local_env_runner, remote_worker_ids=None):
        self.local_env_runner_values.append(local_env_runner)
        self.remote_worker_ids_values.append(remote_worker_ids)
        selected_runners = [self.local_runner] if local_env_runner and self.local_runner else []
        if remote_worker_ids is not None:
            remote_runners = self._remote_runners_for_ids(remote_worker_ids)
        else:
            remote_runners = self._remote_runners_for_ids(self._worker_manager.actor_ids())
        selected_runners.extend(remote_runners)
        return [func(runner) for runner in selected_runners]

    def healthy_worker_ids(self):
        return self._healthy_worker_ids

    def _remote_runners_for_ids(self, actor_ids):
        return [
            runner
            for actor_id, runner in zip(self._worker_manager.actor_ids(), self.runners)
            if actor_id in actor_ids
        ]


class FakeWorkerManager:
    def __init__(self, actor_ids):
        self._actor_ids = actor_ids

    def actor_ids(self):
        return self._actor_ids


class FakeMetrics:
    def __init__(self, eval_results):
        self.eval_results = eval_results
        self.aggregate_calls = []
        self.peek_calls = []

    def aggregate(self, metrics, *, key):
        self.aggregate_calls.append((metrics, key))

    def peek(self, key):
        self.peek_calls.append(key)
        return self.eval_results


class FakeAlgorithm:
    def __init__(self):
        self.iteration = 12
        self.metrics = FakeMetrics({"episode_return_mean": 3.5})


def test_evaluate_and_plot_iv_curve_saves_once_and_returns_metrics(monkeypatch):
    episode = FakeEpisode(
        infos=[
            {
                "arcsinh_huber_loss": 0.00123,
                "i_sim_current_matrix": np.array(
                    [
                        [1.1e-3, 2.1e-3],
                        [2.1e-3, 3.1e-3],
                    ]
                ),
            }
        ],
        env_steps=44,
        agent_steps=55,
    )
    runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"runner_metric": 2},
        env=FakeVectorEnv(FakeEnv(curve_condition_values=[0.1, 0.2])),
    )
    unsampled_runner = FakeEnvRunner(
        episodes=[
            FakeEpisode(
                infos=[
                    {
                        "arcsinh_huber_loss": 99.0,
                        "i_sim_current_matrix": np.array(
                            [
                                [9.1e-3, 9.2e-3],
                                [9.3e-3, 9.4e-3],
                            ]
                        ),
                    }
                ],
                env_steps=1000,
                agent_steps=2000,
            )
        ],
        metrics={"runner_metric": 99},
        env=FakeVectorEnv(FakeEnv(curve_condition_values=[0.1, 0.2])),
    )
    eval_workers = FakeEvalWorkerGroup(runner, unsampled_runner)
    algorithm = FakeAlgorithm()
    save_calls = []

    def fake_save_evaluation_iv_curves(**kwargs):
        save_calls.append(kwargs)
        return {"linear": "linear.png"}

    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        fake_save_evaluation_iv_curves,
    )

    eval_results, env_steps, agent_steps = evaluate_and_plot_iv_curve(
        algorithm,
        eval_workers,
    )

    assert eval_results == {"episode_return_mean": 3.5}
    assert env_steps == 44
    assert agent_steps == 55
    assert runner.sample_calls == [1]
    assert unsampled_runner.sample_calls == []
    assert len(save_calls) == 1
    save_call = save_calls[0]
    assert save_call["curve_condition_values"] == [0.1, 0.2]
    assert save_call["plot_data"]["i_sim_current_matrix"] is episode.infos[-1][
        "i_sim_current_matrix"
    ]
    assert save_call["evaluation_index"] == 1
    assert save_call["training_iteration"] == 12
    assert save_call["fit_loss"] == 0.00123
    assert algorithm._iv_curve_evaluation_index == 1
    metric_key = (EVALUATION_RESULTS, ENV_RUNNER_RESULTS)
    assert algorithm.metrics.aggregate_calls == [([{"runner_metric": 2}], metric_key)]
    assert algorithm.metrics.peek_calls == [metric_key]


def test_evaluate_and_plot_iv_curve_saves_with_local_only_eval_runner(monkeypatch):
    episode = FakeEpisode(
        infos=[
            {
                "arcsinh_huber_loss": 0.00234,
                "i_sim_current_matrix": np.array(
                    [
                        [1.1e-3, 2.1e-3],
                        [2.1e-3, 3.1e-3],
                    ]
                ),
            }
        ],
        env_steps=33,
        agent_steps=44,
    )
    local_runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"local_metric": 7},
        env=FakeEnv(curve_condition_values=None, vds=[0.1, 0.2]),
    )
    eval_workers = FakeEvalWorkerGroup(actor_ids=[], local_runner=local_runner)
    algorithm = FakeAlgorithm()
    save_calls = []
    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        lambda **kwargs: save_calls.append(kwargs),
    )

    eval_results, env_steps, agent_steps = evaluate_and_plot_iv_curve(
        algorithm,
        eval_workers,
    )

    assert eval_results == {"episode_return_mean": 3.5}
    assert env_steps == 33
    assert agent_steps == 44
    assert local_runner.sample_calls == [1]
    assert len(save_calls) == 1
    assert save_calls[0]["curve_condition_values"] == [0.1, 0.2]


def test_evaluate_and_plot_iv_curve_warns_and_skips_save_when_matrix_missing(
    monkeypatch,
    caplog,
):
    episode = FakeEpisode(
        infos=[{"arcsinh_huber_loss": 0.00456}],
        env_steps=11,
        agent_steps=22,
    )
    runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"runner_metric": 2},
        env=FakeEnv(curve_condition_values=[0.1, 0.2]),
    )
    eval_workers = FakeEvalWorkerGroup(runner)
    algorithm = FakeAlgorithm()
    save_calls = []
    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        lambda **kwargs: save_calls.append(kwargs),
    )

    with caplog.at_level(logging.WARNING):
        eval_results, env_steps, agent_steps = evaluate_and_plot_iv_curve(
            algorithm,
            eval_workers,
        )

    assert eval_results == {"episode_return_mean": 3.5}
    assert env_steps == 11
    assert agent_steps == 22
    assert save_calls == []
    assert "i_sim_current_matrix" in caplog.text


def test_evaluate_and_plot_iv_curve_uses_env_matrix_when_episode_info_missing(
    monkeypatch,
):
    episode = FakeEpisode(
        infos=[{"arcsinh_huber_loss": 0.00456}],
        env_steps=11,
        agent_steps=22,
    )
    fallback_matrix = np.array(
        [
            [1.1e-3, 2.1e-3],
            [2.1e-3, 3.1e-3],
        ]
    )
    env = FakeEnv(curve_condition_values=[0.1, 0.2])
    env.last_i_sim_current_matrix = fallback_matrix
    runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"runner_metric": 2},
        env=env,
    )
    eval_workers = FakeEvalWorkerGroup(runner)
    algorithm = FakeAlgorithm()
    save_calls = []
    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        lambda **kwargs: save_calls.append(kwargs),
    )

    eval_results, env_steps, agent_steps = evaluate_and_plot_iv_curve(
        algorithm,
        eval_workers,
    )

    assert eval_results == {"episode_return_mean": 3.5}
    assert env_steps == 11
    assert agent_steps == 22
    assert len(save_calls) == 1
    assert save_calls[0]["plot_data"]["i_sim_current_matrix"] is fallback_matrix


def test_evaluate_and_plot_iv_curve_warns_and_skips_save_when_static_data_missing(
    monkeypatch,
    caplog,
):
    episode = FakeEpisode(
        infos=[
            {
                "arcsinh_huber_loss": 0.00456,
                "i_sim_current_matrix": np.array(
                    [
                        [1.1e-3, 2.1e-3],
                        [2.1e-3, 3.1e-3],
                    ]
                ),
            }
        ],
        env_steps=11,
        agent_steps=22,
    )
    runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"runner_metric": 2},
        env=FakeEnvWithoutPlotData(),
    )
    eval_workers = FakeEvalWorkerGroup(runner)
    algorithm = FakeAlgorithm()
    save_calls = []
    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        lambda **kwargs: save_calls.append(kwargs),
    )

    with caplog.at_level(logging.WARNING):
        eval_results, env_steps, agent_steps = evaluate_and_plot_iv_curve(
            algorithm,
            eval_workers,
        )

    assert eval_results == {"episode_return_mean": 3.5}
    assert env_steps == 11
    assert agent_steps == 22
    assert save_calls == []
    assert "static plot data" in caplog.text


def test_evaluate_and_plot_iv_curve_logs_and_propagates_plot_failure(
    monkeypatch,
    caplog,
):
    episode = FakeEpisode(
        infos=[
            {
                "arcsinh_huber_loss": 0.00456,
                "i_sim_current_matrix": np.array(
                    [
                        [1.1e-3, 2.1e-3],
                        [2.1e-3, 3.1e-3],
                    ]
                ),
            }
        ],
        env_steps=11,
        agent_steps=22,
    )
    runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"runner_metric": 2},
        env=FakeEnv(curve_condition_values=[0.1, 0.2]),
    )
    eval_workers = FakeEvalWorkerGroup(runner)
    algorithm = FakeAlgorithm()

    def raise_plot_failure(**kwargs):
        raise RuntimeError("plot failed")

    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        raise_plot_failure,
    )

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="plot failed"):
        evaluate_and_plot_iv_curve(algorithm, eval_workers)

    assert "Failed to save evaluation I-V curve plot" in caplog.text
    assert "evaluation_index=1" in caplog.text


def test_episode_step_helpers_support_methods_and_attributes():
    method_episode = FakeEpisode(env_steps=7, agent_steps=8)
    attribute_episode = FakeEpisode(env_steps=9, agent_steps=10, callable_steps=False)

    assert _episode_env_steps(method_episode) == 7
    assert _episode_agent_steps(method_episode) == 8
    assert _episode_env_steps(attribute_episode) == 9
    assert _episode_agent_steps(attribute_episode) == 10


def test_episode_final_info_prefers_rllib_get_infos():
    episode = FakeGetInfosEpisode({"i_sim_current_matrix": "matrix"})

    assert _episode_final_info(episode) == {"i_sim_current_matrix": "matrix"}


def test_plot_dir_anchors_relative_plot_dir_to_project_root(monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", "/project")
    monkeypatch.setenv("PLOT_DIR", "result/iv_curve")
    monkeypatch.setenv("ALGO_NAME", "ppo")

    assert _plot_dir().startswith("/project/result/iv_curve/ppo/")


def test_evaluate_and_plot_iv_curve_prefers_episode_best_matrix(monkeypatch):
    final_matrix = np.array(
        [
            [1.1e-3, 2.1e-3],
            [2.1e-3, 3.1e-3],
        ]
    )
    best_matrix = np.array(
        [
            [0.9e-3, 1.9e-3],
            [1.9e-3, 2.9e-3],
        ]
    )
    episode = FakeEpisode(
        infos=[
            {
                "arcsinh_huber_loss": 0.00456,
                "episode_best_arcsinh_huber_loss": 0.00123,
                "i_sim_current_matrix": final_matrix,
                "episode_best_i_sim_current_matrix": best_matrix,
            }
        ],
        env_steps=11,
        agent_steps=22,
    )
    runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"runner_metric": 2},
        env=FakeEnv(curve_condition_values=[0.1, 0.2]),
    )
    eval_workers = FakeEvalWorkerGroup(runner)
    algorithm = FakeAlgorithm()
    save_calls = []
    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        lambda **kwargs: save_calls.append(kwargs),
    )

    evaluate_and_plot_iv_curve(algorithm, eval_workers)

    assert len(save_calls) == 1
    assert save_calls[0]["plot_data"]["i_sim_current_matrix"] is best_matrix
    assert save_calls[0]["fit_loss"] == 0.00123


def test_evaluate_and_plot_iv_curve_prefers_env_episode_best_over_final_matrix(
    monkeypatch,
):
    final_matrix = np.array(
        [
            [1.1e-3, 2.1e-3],
            [2.1e-3, 3.1e-3],
        ]
    )
    env_best_matrix = np.array(
        [
            [0.9e-3, 1.9e-3],
            [1.9e-3, 2.9e-3],
        ]
    )
    episode = FakeEpisode(
        infos=[
            {
                "arcsinh_huber_loss": 0.00456,
                "i_sim_current_matrix": final_matrix,
            }
        ],
        env_steps=11,
        agent_steps=22,
    )
    env = FakeEnv(curve_condition_values=[0.1, 0.2])
    env.episode_best_i_sim_current_matrix = env_best_matrix
    runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"runner_metric": 2},
        env=env,
    )
    eval_workers = FakeEvalWorkerGroup(runner)
    algorithm = FakeAlgorithm()
    save_calls = []
    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        lambda **kwargs: save_calls.append(kwargs),
    )

    evaluate_and_plot_iv_curve(algorithm, eval_workers)

    assert len(save_calls) == 1
    assert save_calls[0]["plot_data"]["i_sim_current_matrix"] is env_best_matrix


def test_evaluate_and_plot_iv_curve_falls_back_when_best_fit_loss_is_none(
    monkeypatch,
):
    matrix = np.array(
        [
            [1.1e-3, 2.1e-3],
            [2.1e-3, 3.1e-3],
        ]
    )
    episode = FakeEpisode(
        infos=[
            {
                "episode_best_arcsinh_huber_loss": None,
                "arcsinh_huber_loss": 0.00456,
                "i_sim_current_matrix": matrix,
            }
        ],
        env_steps=11,
        agent_steps=22,
    )
    runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"runner_metric": 2},
        env=FakeEnv(curve_condition_values=[0.1, 0.2]),
    )
    eval_workers = FakeEvalWorkerGroup(runner)
    algorithm = FakeAlgorithm()
    save_calls = []
    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        lambda **kwargs: save_calls.append(kwargs),
    )

    evaluate_and_plot_iv_curve(algorithm, eval_workers)

    assert len(save_calls) == 1
    assert save_calls[0]["fit_loss"] == 0.00456
