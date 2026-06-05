from env.eehemt_env import key_params_names
from utils.callbacks import TrainingMetricsCallback


class FakeMetricsLogger:
    def __init__(self):
        self.logged_values = []

    def log_value(self, key, value, *, reduce):
        self.logged_values.append((key, value, reduce))


class FakeEpisode:
    def __init__(self, info):
        self.infos = [info]
        self.custom_data = {name: [] for name in key_params_names}


def test_training_metrics_callback_logs_final_nrmse():
    callback = TrainingMetricsCallback()
    metrics_logger = FakeMetricsLogger()
    episode = FakeEpisode(
        {
            "arcsinh_huber_loss": 1.23e-4,
            "nrmse": 4.56,
        }
    )

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=metrics_logger,
        env=None,
        env_index=0,
        rl_module=None,
    )

    assert ("last_nrmse", 4.56, "mean") in metrics_logger.logged_values
    assert ("min_nrmse", 4.56, "mean") in metrics_logger.logged_values
