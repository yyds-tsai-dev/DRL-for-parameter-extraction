from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Only available on Linux with Python 3.11
import verilogae  # type: ignore[import-untyped]


class EEHEMTModelHelper:
    """Utility class for loading EEHEMT model, simulating Ids, printing params, and plotting I-V curves."""

    def __init__(
        self,
        va_file_path: str,
        save_dir: str = "result/iv_curve/manual",
        temperature: int = 300,
        curve_condition_name: str | None = None,
    ) -> None:
        self.va_file_path = str(Path(va_file_path))
        self.save_dir = Path(save_dir)
        self.temperature = temperature

        default_curve_name = os.getenv("CURVE_CONDITION_NAMES", "UGW,NOF").split(",")[0]
        self.curve_condition_name = curve_condition_name or default_curve_name.strip()

        if not Path(self.va_file_path).exists():
            raise FileNotFoundError(f"VA file not found: {self.va_file_path}")

        self.eehemt_model = verilogae.load(self.va_file_path)  # type: ignore
        self._modelcard = {
            name: float(param.default)
            for name, param in self.eehemt_model.modelcard.items()
        }
        self._loaded_init_param_keys: set[str] = set()
        self._modelcard_sync_warned = False

    def _sync_single_param(self, name: str, value: float) -> None:
        """Synchronize one parameter to internal modelcard and VA model if writable."""
        self._modelcard[name] = value

        try:
            self.eehemt_model.modelcard[name].default = value
        except (AttributeError, TypeError):
            if not self._modelcard_sync_warned:
                print(
                    "Warning: eehemt_model.modelcard default is read-only; "
                    "using internal parameter cache for simulation."
                )
                self._modelcard_sync_warned = True

    def _get_modelcard(self) -> dict[str, float]:
        """Return a copy of current modelcard default parameters."""
        return self._modelcard.copy()

    def print_params(self, param_names: list[str] | None = None) -> None:
        """Print selected parameters. If param_names is None, print all parameters."""
        modelcard = self._get_modelcard()
        names = param_names if param_names is not None else sorted(modelcard.keys())

        for name in names:
            if name in modelcard:
                print(f"{name:>12s} = {modelcard[name]:.6g}")
            else:
                print(f"{name:>12s} = <not found>")

    def load_init_params_from_json(self, json_path: str) -> dict[str, float]:
        """Load parameter init values from JSON and return a modelcard update dict."""
        config_path = Path(json_path)
        if not config_path.exists():
            raise FileNotFoundError(f"JSON config not found: {json_path}")

        with config_path.open("r", encoding="utf-8") as f:
            param_cfg = json.load(f)

        init_params: dict[str, float] = {}
        for name, cfg in param_cfg.items():
            if isinstance(cfg, dict) and "init" in cfg and name in self._modelcard:
                init_params[name] = float(cfg["init"])

        if not init_params:
            raise ValueError(
                "No valid 'init' values found in JSON for current modelcard"
            )
        # Synchronize loaded parameters to both modelcard instances
        for name, value in init_params.items():
            self._sync_single_param(name, value)
        self._loaded_init_param_keys = set(init_params.keys())

        return init_params

    def update_modelcard(self, modelcard_updates: dict[str, float]) -> None:
        """Update modelcard parameters in both self._modelcard and self.eehemt_model.modelcard.

        Args:
            modelcard_updates: Dictionary of parameter names and their new values.

        Raises:
            KeyError: If any parameter name is not found in the current modelcard.
        """
        # Validate all keys exist
        invalid_keys = set(modelcard_updates.keys()) - set(self._modelcard.keys())
        if invalid_keys:
            raise KeyError(f"Parameter(s) not found in modelcard: {invalid_keys}")

        # Update both modelcard instances
        for name, value in modelcard_updates.items():
            value_float = float(value)
            self._sync_single_param(name, value_float)

        print(f"Updated {len(modelcard_updates)} parameter(s) in modelcard.")

    def save_params_to_json(
        self, output_path: str, reference_json_path: str | None = None
    ) -> str:
        """Save currently loaded init parameters to a JSON file.

        Only parameters loaded by load_init_params_from_json are saved.
        If reference_json_path is provided, preserves min/max values from the reference file
        and updates only the 'init' values. Otherwise, saves only the 'init' values.

        Args:
            output_path: Path to save the JSON file.
            reference_json_path: Optional path to a reference JSON file (e.g., modelcard.json)
                                 to preserve min/max metadata. If None, only init values are saved.

        Returns:
            str: The absolute path to the saved JSON file.
        """
        if not self._loaded_init_param_keys:
            raise ValueError(
                "No loaded init parameters found. "
                "Call load_init_params_from_json() before saving."
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        loaded_keys = self._loaded_init_param_keys

        # Prepare output dictionary
        if reference_json_path:
            ref_path = Path(reference_json_path)
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Reference JSON not found: {reference_json_path}"
                )

            with ref_path.open("r", encoding="utf-8") as f:
                param_cfg = json.load(f)

            # Update init values while preserving min/max
            for name in param_cfg.keys():
                if (
                    name in loaded_keys
                    and name in self._modelcard
                    and isinstance(param_cfg[name], dict)
                ):
                    param_cfg[name]["init"] = self._modelcard[name]
            param_cfg = {k: v for k, v in param_cfg.items() if k in loaded_keys}
        else:
            # Save only init values
            param_cfg = {
                name: {"init": self._modelcard[name]} for name in sorted(loaded_keys)
            }

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(param_cfg, f, indent=4)

        print(f"Saved {len(param_cfg)} parameter(s) to: {output_file}")
        return str(output_file.resolve())

    def _resolve_ids_function(self):
        """Resolve Ids function name for different VA model implementations."""
        for fn_name in ("Ids", "I_ds", "I_dsat"):
            if fn_name in self.eehemt_model.functions:
                return self.eehemt_model.functions[fn_name]

        available = ", ".join(sorted(self.eehemt_model.functions.keys()))
        raise KeyError(f"No Ids-like function found. Available functions: {available}")

    def _simulate_ids(
        self,
        vgs: np.ndarray,
        vds_voltage: float = 0.5,
        curve_condition_value: float | None = None,
        modelcard_updates: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Simulate Ids over Vgs at fixed Vds and optional curve condition value."""
        vgs_array = np.asarray(vgs, dtype=float)
        vds_array = np.full_like(vgs_array, vds_voltage, dtype=float)
        sweep_bias = {"br_gisi": vgs_array, "br_disi": vds_array}

        sim_params: dict[str, float] = self._get_modelcard()
        if modelcard_updates:
            sim_params.update({k: float(v) for k, v in modelcard_updates.items()})

        if (
            curve_condition_value is not None
            and self.curve_condition_name
            and self.curve_condition_name in sim_params
        ):
            sim_params[self.curve_condition_name] = float(curve_condition_value)

        ids_fn = self._resolve_ids_function()
        i_sim = ids_fn.eval(
            temperature=self.temperature,
            voltages=sweep_bias,
            **sim_params,
        )

        return np.asarray(i_sim, dtype=float)

    def simulate_ids_from_json_init(
        self,
        json_path: str,
        vgs: np.ndarray,
        vds_voltage: float = 0.5,
        curve_condition_value: float | None = None,
        modelcard_updates: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Simulate Ids by applying init values from JSON, then optional overrides."""
        init_updates = self.load_init_params_from_json(json_path)
        merged_updates = init_updates.copy()
        if modelcard_updates:
            merged_updates.update({k: float(v) for k, v in modelcard_updates.items()})

        return self._simulate_ids(
            vgs=vgs,
            vds_voltage=vds_voltage,
            curve_condition_value=curve_condition_value,
            modelcard_updates=merged_updates,
        )

    def plot_iv_curve(
        self,
        vgs: np.ndarray,
        i_sim: np.ndarray,
        i_meas: np.ndarray | None = None,
        title: str = "EEHEMT I-V Curve",
        save_name: str | None = None,
        log_y: bool = False,
    ) -> str:
        """Plot target I-V curve (and optional measured curve) and save figure."""
        vgs_array = np.asarray(vgs, dtype=float)
        i_sim_array = np.asarray(i_sim, dtype=float)

        if len(vgs_array) != len(i_sim_array):
            raise ValueError("vgs and i_sim must have the same length")

        i_meas_array: np.ndarray | None = None
        if i_meas is not None:
            i_meas_array = np.asarray(i_meas, dtype=float)
            if len(i_meas_array) != len(vgs_array):
                raise ValueError("vgs and i_meas must have the same length")

        self.save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            suffix = "log" if log_y else "linear"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"iv_curve_{suffix}_{timestamp}.png"

        save_path = self.save_dir / save_name

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title(title)
        ax.set_xlabel("Gate Voltage (V)")
        ax.set_ylabel("Drain Current (A)")

        if i_meas_array is not None:
            ax.plot(vgs_array, i_meas_array, "o", ms=4, label="Measured")
        ax.plot(vgs_array, i_sim_array, "-", lw=2, label="Target")

        if log_y:
            ax.set_yscale("log")

        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.legend(loc="best")

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot to: {save_path}")
        return str(save_path)

    def simulate_and_plot_from_json_init(
        self,
        json_path: str,
        vgs: np.ndarray,
        vds_voltage: float = 0.5,
        curve_condition_value: float | None = None,
        title: str = "EEHEMT Target I-V Curve",
        save_name: str | None = None,
        log_y: bool = False,
        modelcard_updates: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, str]:
        """Load init params from JSON, simulate Ids, then save target I-V plot."""
        i_sim = self.simulate_ids_from_json_init(
            json_path=json_path,
            vgs=vgs,
            vds_voltage=vds_voltage,
            curve_condition_value=curve_condition_value,
            modelcard_updates=modelcard_updates,
        )
        save_path = self.plot_iv_curve(
            vgs=vgs,
            i_sim=i_sim,
            i_meas=None,
            title=title,
            save_name=save_name,
            log_y=log_y,
        )
        return i_sim, save_path
