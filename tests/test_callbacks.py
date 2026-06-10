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


def test_training_metrics_callback_logs_episode_best_nrmse():
    callback = TrainingMetricsCallback()
    metrics_logger = FakeMetricsLogger()
    episode = FakeEpisode(
        {
            "arcsinh_huber_loss": 1.23e-4,
            "episode_best_arcsinh_huber_loss": 9.87e-5,
            "nrmse": 8.9,
            "episode_best_nrmse": 4.56,
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

    assert ("episode_best_arcsinh_huber_loss", 9.87e-5, "mean") in (
        metrics_logger.logged_values
    )
    assert ("min_arcsinh_huber_loss", 9.87e-5, "min") in (
        metrics_logger.logged_values
    )
    assert ("episode_best_nrmse", 4.56, "mean") in metrics_logger.logged_values
    assert ("min_nrmse", 4.56, "min") in metrics_logger.logged_values
    assert not any(key == "last_nrmse" for key, _, _ in metrics_logger.logged_values)


def test_training_metrics_callback_falls_back_to_final_nrmse_for_old_infos():
    callback = TrainingMetricsCallback()
    metrics_logger = FakeMetricsLogger()
    episode = FakeEpisode(
        {
            "arcsinh_huber_loss": 1.23e-4,
            "nrmse": 7.89,
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

    assert ("episode_best_arcsinh_huber_loss", 1.23e-4, "mean") in (
        metrics_logger.logged_values
    )
    assert ("min_arcsinh_huber_loss", 1.23e-4, "min") in (
        metrics_logger.logged_values
    )
    assert ("episode_best_nrmse", 7.89, "mean") in metrics_logger.logged_values
    assert ("min_nrmse", 7.89, "min") in metrics_logger.logged_values
