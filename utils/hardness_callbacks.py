from __future__ import annotations

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class HardnessMetricsCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.max_predicted_hardness = float("-inf")
        self.best_composition: dict[str, float] | None = None

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
        predicted_hardness = last_info.get("predicted_hardness")
        if predicted_hardness is None:
            return

        predicted_hardness = float(predicted_hardness)
        uncertainty_hardness = float(last_info.get("uncertainty_hardness", 0.0))
        is_success = bool(last_info.get("is_success", False))
        if predicted_hardness > self.max_predicted_hardness:
            self.max_predicted_hardness = predicted_hardness
            self.best_composition = dict(last_info.get("composition", {}))

        metrics_logger.log_value("predicted_hardness", predicted_hardness, reduce="mean")
        metrics_logger.log_value(
            "max_predicted_hardness", self.max_predicted_hardness, reduce="max"
        )
        metrics_logger.log_value(
            "uncertainty_hardness", uncertainty_hardness, reduce="mean"
        )
        metrics_logger.log_value(
            "success_rate_650", 1.0 if is_success else 0.0, reduce="mean"
        )
