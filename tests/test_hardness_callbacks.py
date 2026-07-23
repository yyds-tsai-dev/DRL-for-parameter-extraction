from utils.hardness_callbacks import HardnessMetricsCallback


class FakeMetricsLogger:
    def __init__(self):
        self.values = {}

    def log_value(self, key, value, reduce=None):
        self.values[key] = (value, reduce)


class FakeEpisode:
    def __init__(self, infos):
        self.infos = infos


def test_hardness_callback_logs_max_hardness_and_success_rate():
    callback = HardnessMetricsCallback()
    metrics_logger = FakeMetricsLogger()
    episode = FakeEpisode(
        [
            {
                "predicted_hardness": 720.0,
                "uncertainty_hardness": 11.0,
                "composition": {"frac_Al": 0.2, "frac_Ni": 0.2},
                "is_success": True,
            }
        ]
    )

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=metrics_logger,
        env=None,
        env_index=0,
        rl_module=None,
    )

    assert metrics_logger.values["predicted_hardness"] == (720.0, "mean")
    assert metrics_logger.values["max_predicted_hardness"] == (720.0, "max")
    assert metrics_logger.values["uncertainty_hardness"] == (11.0, "mean")
    assert metrics_logger.values["success_rate_650"] == (1.0, "mean")
    assert callback.best_composition == {"frac_Al": 0.2, "frac_Ni": 0.2}


def test_hardness_callback_keeps_global_max():
    callback = HardnessMetricsCallback()
    metrics_logger = FakeMetricsLogger()

    for hardness in (700.0, 680.0):
        episode = FakeEpisode(
            [
                {
                    "predicted_hardness": hardness,
                    "uncertainty_hardness": 1.0,
                    "composition": {"frac_Ni": hardness},
                    "is_success": hardness >= 650.0,
                }
            ]
        )
        callback.on_episode_end(
            episode=episode,
            env_runner=None,
            metrics_logger=metrics_logger,
            env=None,
            env_index=0,
            rl_module=None,
        )

    assert metrics_logger.values["max_predicted_hardness"] == (700.0, "max")
    assert callback.best_composition == {"frac_Ni": 700.0}


def test_hardness_callback_logs_threshold_agnostic_success_rate():
    callback = HardnessMetricsCallback()
    metrics_logger = FakeMetricsLogger()
    episode = FakeEpisode(
        [
            {
                "predicted_hardness": 700.0,
                "uncertainty_hardness": 2.0,
                "composition": {"frac_Ni": 0.2},
                "is_success": True,
            }
        ]
    )

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=metrics_logger,
        env=None,
        env_index=0,
        rl_module=None,
    )

    assert metrics_logger.values["success_rate"] == (1.0, "mean")
    assert metrics_logger.values["success_rate_650"] == (1.0, "mean")
