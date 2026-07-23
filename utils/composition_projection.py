from __future__ import annotations

import numbers

import numpy as np


def _finite_float(name: str, value) -> float:
    scalar_array = np.asarray(value)
    if scalar_array.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar")

    scalar_value = scalar_array.item()
    if not isinstance(scalar_value, (numbers.Real, np.number)):
        raise ValueError(f"{name} must be a finite scalar")

    try:
        scalar = float(scalar_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def project_bounded_simplex(
    values,
    *,
    lower: float,
    upper: float,
    target_sum: float,
    atol: float = 1e-10,
) -> np.ndarray:
    """Project values to {x | lower <= x_i <= upper, sum(x) = target_sum}."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("values must be a one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must contain only finite numbers")

    lower = _finite_float("lower", lower)
    upper = _finite_float("upper", upper)
    target_sum = _finite_float("target_sum", target_sum)
    atol = _finite_float("atol", atol)
    if atol < 0.0:
        raise ValueError("atol must be non-negative")
    if lower > upper:
        raise ValueError("lower must be less than or equal to upper")

    size = vector.size
    min_sum = lower * size
    max_sum = upper * size
    min_sum_boundary = np.nextafter(min_sum, -np.inf)
    max_sum_boundary = np.nextafter(max_sum, np.inf)
    if target_sum < min_sum_boundary or target_sum > max_sum_boundary:
        raise ValueError(
            "target_sum is outside feasible bounds: "
            f"{target_sum} not in [{min_sum}, {max_sum}]"
        )

    if size == 0:
        return np.array([], dtype=np.float32)

    order = np.argsort(vector)[::-1]
    sorted_values = vector[order]

    for upper_count in range(size + 1):
        for lower_count in range(size - upper_count + 1):
            free_start = upper_count
            free_end = size - lower_count
            free_count = free_end - free_start
            fixed_sum = upper_count * upper + lower_count * lower
            free_sum = target_sum - fixed_sum

            if free_count == 0:
                if abs(free_sum) > atol:
                    continue
                sorted_projected = np.empty(size, dtype=np.float64)
                sorted_projected[:upper_count] = upper
                sorted_projected[free_end:] = lower
                projected = np.empty(size, dtype=np.float64)
                projected[order] = sorted_projected
                return projected.astype(np.float32)

            if free_sum < lower * free_count - atol:
                continue
            if free_sum > upper * free_count + atol:
                continue

            free_values = sorted_values[free_start:free_end]
            center = float(free_values[0] / 2.0 + free_values[-1] / 2.0)
            shifted = free_values - center
            theta = (float(np.sum(shifted, dtype=np.float64)) - free_sum) / free_count
            free_projected = shifted - theta

            if np.any(free_projected < lower - atol):
                continue
            if np.any(free_projected > upper + atol):
                continue

            if upper_count:
                upper_projected = sorted_values[:upper_count] - center - theta
                if np.any(upper_projected < upper - atol):
                    continue
            if lower_count:
                lower_projected = sorted_values[free_end:] - center - theta
                if np.any(lower_projected > lower + atol):
                    continue

            sorted_projected = np.empty(size, dtype=np.float64)
            sorted_projected[:upper_count] = upper
            sorted_projected[free_start:free_end] = np.clip(free_projected, lower, upper)
            sorted_projected[free_end:] = lower

            projected = np.empty(size, dtype=np.float64)
            projected[order] = sorted_projected
            return projected.astype(np.float32)

    raise RuntimeError("failed to project values to the bounded simplex")
