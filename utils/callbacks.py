import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from env.eehemt_env import key_params_names
from utils.logging_config import get_logger

logger = get_logger(__name__)


class TrainingMetricsCallback(DefaultCallbacks):
    """
    RLlib callback for recording training metrics at episode boundaries.
    """

    def __init__(self):
        super().__init__()
        self.min_arcsinh_huber_loss = float("inf")
        self.min_nrmse = float("inf")

    def on_episode_start(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        # episode.custom_data["Vto"] = []
        tunable_params = {name: [] for name in key_params_names}
        episode.custom_data.update(tunable_params)  # type: ignore

    def on_episode_step(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        callback_env = env if env is not None else base_env
        envs = getattr(callback_env, "envs", None)
        actual_env = envs[env_index if env_index is not None else 0] if envs else callback_env
        actual_env = getattr(actual_env, "unwrapped", actual_env)
        current_params = actual_env.current_params  # type: ignore
        # print(f"current_params: {current_params}")
        # episode.custom_data["Vto"].append(current_params["Vto"])
        for param_name in key_params_names:
            episode.custom_data[param_name].append(current_params[param_name])  # type: ignore

    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        infos = getattr(episode, "infos", None) or []
        last_info = infos[-1] if infos else {}
        fit_loss = last_info.get(
            "episode_best_arcsinh_huber_loss",
            last_info.get("arcsinh_huber_loss"),
        )
        if fit_loss is not None:
            if fit_loss < self.min_arcsinh_huber_loss:
                self.min_arcsinh_huber_loss = fit_loss

            metrics_logger.log_value(
                "episode_best_arcsinh_huber_loss",
                fit_loss,
                reduce="mean",
            )
            metrics_logger.log_value(
                "min_arcsinh_huber_loss",
                self.min_arcsinh_huber_loss,
                reduce="mean",
            )
            logger.info(
                "Episode-best arcsinh Huber loss: %.6g; "
                "Min arcsinh Huber loss: %.6g",
                fit_loss,
                self.min_arcsinh_huber_loss,
            )
        nrmse = last_info.get("episode_best_nrmse", last_info.get("nrmse"))
        if nrmse is not None:
            if nrmse < self.min_nrmse:
                self.min_nrmse = nrmse

            metrics_logger.log_value(
                "episode_best_nrmse",
                nrmse,
                reduce="mean",
            )
            metrics_logger.log_value(
                "min_nrmse",
                self.min_nrmse,
                reduce="mean",
            )
            logger.info(
                "Episode-best NRMSE: %.6g; Min NRMSE: %.6g",
                nrmse,
                self.min_nrmse,
            )

        for param_name in key_params_names:
            param_values = episode.custom_data[param_name]

            avg_param_value = np.mean(param_values) if param_values else 0.0

            log_key = f"avg_{param_name}"

            metrics_logger.log_value(
                log_key,
                avg_param_value,
                reduce="mean",
            )
        return


PlotCurve = TrainingMetricsCallback
