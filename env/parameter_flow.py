from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# Only available on Linux with Python 3.11.
import verilogae  # type: ignore[import-untyped]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    min_value: float
    max_value: float
    factor: float

    @classmethod
    def from_mapping(cls, name: str, config: Mapping[str, float]) -> "ParameterSpec":
        return cls(
            name=name,
            min_value=float(config["min"]),
            max_value=float(config["max"]),
            factor=float(config.get("factor", 0.0)),
        )

    def clamp(self, value: float) -> float:
        return float(np.clip(value, self.min_value, self.max_value))


class ParameterSpecCollection:
    def __init__(self, specs: Sequence[ParameterSpec]) -> None:
        self._specs = list(specs)
        self._by_name = {spec.name: spec for spec in self._specs}
        self.names = [spec.name for spec in self._specs]

    @classmethod
    def from_config(
        cls,
        all_possible_params: Mapping[str, Mapping[str, float]],
        selected_names: Sequence[str],
    ) -> "ParameterSpecCollection":
        return cls(
            [
                ParameterSpec.from_mapping(name, all_possible_params[name])
                for name in selected_names
                if name in all_possible_params
            ]
        )

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def min_values(self) -> np.ndarray:
        return np.array([spec.min_value for spec in self._specs], dtype=np.float32)

    @property
    def max_values(self) -> np.ndarray:
        return np.array([spec.max_value for spec in self._specs], dtype=np.float32)

    def action_factors(self, range_fraction: float = 0.01) -> np.ndarray:
        return np.array(
            [
                (spec.max_value - spec.min_value) * range_fraction
                for spec in self._specs
            ],
            dtype=np.float32,
        )

    def ensure_control_defaults(self, params: Mapping[str, float]) -> dict[str, float]:
        normalized = {name: float(value) for name, value in params.items()}
        for spec in self._specs:
            normalized.setdefault(spec.name, spec.min_value)
        return normalized

    def normalize_params(self, params: Mapping[str, float]) -> dict[str, float]:
        normalized = self.ensure_control_defaults(params)
        for spec in self._specs:
            normalized[spec.name] = spec.clamp(normalized[spec.name])
        self.apply_derived_constraints(normalized)
        return normalized

    def apply_delta(
        self,
        params: Mapping[str, float],
        delta: Sequence[float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        updated = self.ensure_control_defaults(params)
        actual_delta: dict[str, float] = {}
        for index, spec in enumerate(self._specs):
            before = updated[spec.name]
            after = spec.clamp(before + float(delta[index]))
            updated[spec.name] = after
            actual_delta[spec.name] = after - before
        self.apply_derived_constraints(updated)
        return updated, actual_delta

    @staticmethod
    def apply_derived_constraints(params: dict[str, float]) -> None:
        if "Vco" in params and "DVcoVgo" in params:
            params["Vgo"] = params["Vco"] - params["DVcoVgo"]
        if all(name in params for name in ("Vco", "DVcoVgo", "DVgoVto")):
            params["Vto"] = params["Vco"] - params["DVcoVgo"] - params["DVgoVto"]
        if "Vtso" in params and "DVtsoVto" in params:
            params["Vto"] = params["Vtso"] - params["DVtsoVto"]


@dataclass(frozen=True)
class MeasuredCurveDataset:
    vgs: np.ndarray
    vds: list[float]
    current_by_vds: dict[float, np.ndarray]
    current_matrix: np.ndarray

    @classmethod
    def from_csv(
        cls,
        csv_file_path: str | Path,
        *,
        primary_vds_count: int = 10,
        default_extra_vds: Sequence[float] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5),
    ) -> "MeasuredCurveDataset":
        measured_df = pd.read_csv(csv_file_path)
        if "vg" not in measured_df.columns:
            raise ValueError("Measured curve CSV must contain a 'vg' column")

        numeric_vds_columns = [
            float(column) for column in measured_df.columns if column != "vg"
        ]
        selected_vds = numeric_vds_columns[1 : primary_vds_count + 1]
        for vd_value in default_extra_vds:
            if vd_value in numeric_vds_columns and vd_value not in selected_vds:
                selected_vds.append(float(vd_value))

        current_by_vds = {
            vd_value: measured_df[str(vd_value)].to_numpy(dtype=np.float64)
            for vd_value in selected_vds
        }
        current_matrix = np.array(
            [current_by_vds[vd_value] for vd_value in selected_vds],
            dtype=np.float64,
        )
        return cls(
            vgs=measured_df["vg"].to_numpy(dtype=np.float64),
            vds=selected_vds,
            current_by_vds=current_by_vds,
            current_matrix=current_matrix,
        )


class EEHEMTSimulator:
    def __init__(
        self,
        eehemt_model,
        *,
        temperature: int,
        rs_ext: float,
        rd_ext: float,
        ir_drop_n_iter: int,
        ir_drop_maxfev: int,
    ) -> None:
        self.eehemt_model = eehemt_model
        self.temperature = temperature
        self.rs_ext = rs_ext
        self.rd_ext = rd_ext
        self.ir_drop_n_iter = ir_drop_n_iter
        self.ir_drop_maxfev = ir_drop_maxfev
        self.ir_drop_residual_tol = float(os.getenv("IR_DROP_RESIDUAL_TOL", 1e-8))
        self.last_solver_diagnostics: list[dict[str, object]] = []

    @classmethod
    def from_va_file(
        cls,
        va_file_path: str,
        *,
        temperature: int,
        rs_ext: float,
        rd_ext: float,
        ir_drop_n_iter: int,
        ir_drop_maxfev: int,
    ) -> "EEHEMTSimulator":
        return cls(
            verilogae.load(va_file_path),  # type: ignore
            temperature=temperature,
            rs_ext=rs_ext,
            rd_ext=rd_ext,
            ir_drop_n_iter=ir_drop_n_iter,
            ir_drop_maxfev=ir_drop_maxfev,
        )

    def modelcard_defaults(self) -> dict[str, float]:
        return {
            name: float(param.default)
            for name, param in self.eehemt_model.modelcard.items()
        }

    def simulate_current_matrix(
        self,
        *,
        params: Mapping[str, float],
        vgs: np.ndarray,
        vds_values: Sequence[float],
        current_step: int = 0,
    ) -> np.ndarray:
        i_sim_results = []
        self.last_solver_diagnostics = []
        sim_params = {k: float(v) for k, v in params.items()}
        for synthetic_param in ("DVcoVgo", "DVgoVto", "DVtsoVto"):
            sim_params.pop(synthetic_param, None)

        rs_total = float(sim_params["Rs"]) + self.rs_ext
        rd_total = float(sim_params["Rd"]) + self.rd_ext
        i_total_eval = self.eehemt_model.functions["I_total"].eval
        vgs_array = np.asarray(vgs, dtype=np.float64)

        previous_solution: np.ndarray | None = None
        zero_start = np.zeros(len(vgs_array), dtype=np.float64)

        for vd_app in vds_values:
            warmup_start = zero_start.copy()
            for _ in range(self.ir_drop_n_iter):
                warmup_start = self._evaluate_current(
                    i_total_eval=i_total_eval,
                    sim_params=sim_params,
                    vgs=vgs_array,
                    vd_app=float(vd_app),
                    current_guess=warmup_start,
                    rs_total=rs_total,
                    rd_total=rd_total,
                )

            def _ir_drop_residual(i_est_iter: np.ndarray) -> np.ndarray:
                i_model = self._evaluate_current(
                    i_total_eval=i_total_eval,
                    sim_params=sim_params,
                    vgs=vgs_array,
                    vd_app=float(vd_app),
                    current_guess=i_est_iter,
                    rs_total=rs_total,
                    rd_total=rd_total,
                )
                return i_est_iter - i_model

            candidates: list[tuple[str, np.ndarray]] = []
            if previous_solution is not None:
                candidates.append(("continuation", previous_solution.copy()))
            candidates.append(("zero", zero_start.copy()))
            if not any(
                np.allclose(warmup_start, candidate, rtol=1e-12, atol=1e-15)
                for _, candidate in candidates
            ):
                candidates.append(("warmup", warmup_start.copy()))

            selected_solution: np.ndarray | None = None
            selected_attempt: dict[str, object] | None = None
            attempts: list[dict[str, object]] = []

            for start_name, start_value in candidates:
                solution, _, ier, msg = fsolve(
                    func=_ir_drop_residual,
                    x0=start_value,
                    maxfev=self.ir_drop_maxfev,
                    full_output=True,
                )
                solution = np.nan_to_num(
                    solution, nan=0.0, posinf=0.1, neginf=-0.1
                )
                residual = _ir_drop_residual(solution)
                residual_max_abs = float(np.max(np.abs(residual)))
                accepted = ier == 1 or residual_max_abs <= self.ir_drop_residual_tol
                attempt = {
                    "start": start_name,
                    "ier": int(ier),
                    "message": str(msg).strip(),
                    "residual_max_abs": residual_max_abs,
                    "accepted": accepted,
                }
                attempts.append(attempt)
                if selected_attempt is None or residual_max_abs < float(
                    selected_attempt["residual_max_abs"]
                ):
                    selected_attempt = attempt
                    selected_solution = solution
                if accepted:
                    selected_attempt = attempt
                    selected_solution = solution
                    break

            if selected_solution is None or selected_attempt is None:
                raise RuntimeError("IR-drop solver did not run any attempts")

            i_sim_single_curve = selected_solution
            converged = bool(selected_attempt["accepted"])
            if converged:
                previous_solution = i_sim_single_curve.copy()
            diagnostic = {
                "vds": float(vd_app),
                "converged": converged,
                "ier": int(selected_attempt["ier"]),
                "message": str(selected_attempt["message"]),
                "residual_max_abs": float(selected_attempt["residual_max_abs"]),
                "selected_start": str(selected_attempt["start"]),
                "attempts": attempts,
            }
            self.last_solver_diagnostics.append(diagnostic)
            if not converged:
                print(
                    f"Warning: IR-drop solver non-converged at step={current_step}, "
                    f"Vds={vd_app:.4g}, ier={selected_attempt['ier']}, "
                    f"residual_max_abs={selected_attempt['residual_max_abs']:.3e}, "
                    f"message={selected_attempt['message']}"
                )

            i_sim_results.append(
                np.nan_to_num(i_sim_single_curve, nan=0.0, posinf=0.1, neginf=-0.1)
            )

        return np.array(i_sim_results, dtype=np.float64)

    def _evaluate_current(
        self,
        *,
        i_total_eval,
        sim_params: Mapping[str, float],
        vgs: np.ndarray,
        vd_app: float,
        current_guess: np.ndarray,
        rs_total: float,
        rd_total: float,
    ) -> np.ndarray:
        vs_node = current_guess * rs_total
        vd_node = vd_app - current_guess * rd_total
        vgs_int = vgs - vs_node
        vds_int = vd_node - vs_node
        vgd_int = vgs_int - vds_int
        return np.asarray(
            i_total_eval(
                temperature=self.temperature,
                voltages={
                    "br_gisi": vgs_int,
                    "br_disi": vds_int,
                    "br_gidi": vgd_int,
                },
                **sim_params,
            ),
            dtype=np.float64,
        ).ravel()


class ArcsinhHuberMetric:
    def __init__(self, *, delta: float, epsilon: float) -> None:
        self.delta = delta
        self.epsilon = epsilon

    def loss_matrix(self, measured: np.ndarray, simulated: np.ndarray) -> np.ndarray:
        diff = np.arcsinh(simulated) - np.arcsinh(measured)
        abs_diff = np.abs(diff)
        return np.where(
            abs_diff <= self.delta,
            0.5 * diff**2,
            self.delta * (abs_diff - 0.5 * self.delta),
        )

    def loss(self, measured: np.ndarray, simulated: np.ndarray) -> float:
        return float(np.mean(self.loss_matrix(measured, simulated)))

    def per_curve_loss(self, measured: np.ndarray, simulated: np.ndarray) -> np.ndarray:
        return np.mean(self.loss_matrix(measured, simulated), axis=1).astype(np.float32)

    def reward(
        self,
        measured: np.ndarray,
        simulated: np.ndarray,
        *,
        reward_min: float,
        reward_max: float,
    ) -> float:
        return self.scaled_reward_from_loss(
            self.loss(measured, simulated),
            reward_min=reward_min,
            reward_max=reward_max,
        )

    def scaled_reward_from_loss(
        self,
        loss: float,
        *,
        reward_min: float,
        reward_max: float,
    ) -> float:
        reward = -np.log10(float(loss) + self.epsilon)
        return float(np.clip(reward, reward_min, reward_max))

    def is_success(
        self,
        measured: np.ndarray,
        simulated: np.ndarray,
        *,
        threshold: float,
    ) -> bool:
        return self.loss(measured, simulated) < threshold
