import numpy as np

from utils.composition_projection import project_bounded_simplex


def test_project_bounded_simplex_preserves_feasible_vector():
    vector = np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15], dtype=np.float64)

    projected = project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=1.0)

    assert projected.dtype == np.float32
    assert np.allclose(projected, vector)
    assert np.isclose(projected.sum(), 1.0)


def test_project_bounded_simplex_enforces_bounds_and_sum():
    vector = np.array([2.0, -1.0, 0.3, 0.1, 0.8, -0.5], dtype=np.float64)

    projected = project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=1.0)

    assert projected.shape == (6,)
    assert np.all(projected >= 0.05 - 1e-8)
    assert np.all(projected <= 0.35 + 1e-8)
    assert np.isclose(projected.sum(), 1.0, atol=1e-8)


def test_project_bounded_simplex_handles_high_magnitude_equal_values():
    vector = np.full(6, 1e16, dtype=np.float64)

    projected = project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=1.0)

    assert projected.shape == (6,)
    assert np.all(projected >= 0.05 - 1e-8)
    assert np.all(projected <= 0.35 + 1e-8)
    assert np.isclose(projected.sum(), 1.0, atol=1e-8)


def test_project_bounded_simplex_handles_high_magnitude_large_spread_values():
    vector = np.array([1e32, -1e32, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    expected = np.array([0.35, 0.05, 0.15, 0.15, 0.15, 0.15], dtype=np.float32)

    projected = project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=1.0)

    assert np.allclose(projected, expected, atol=1e-8)
    assert np.isclose(projected.sum(), 1.0, atol=1e-8)


def test_project_bounded_simplex_rejects_infeasible_bounds():
    vector = np.zeros(6, dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=0.2, upper=0.35, target_sum=1.0)
    except ValueError as exc:
        assert "target_sum is outside feasible bounds" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_project_bounded_simplex_accepts_exact_decimal_minimum_sum():
    vector = np.zeros(6, dtype=np.float64)

    projected = project_bounded_simplex(vector, lower=0.1, upper=0.9, target_sum=0.6)

    assert np.allclose(projected, np.full(6, 0.1, dtype=np.float32))
    assert np.isclose(projected.sum(), 0.6, atol=1e-8)


def test_project_bounded_simplex_accepts_exact_decimal_maximum_sum():
    vector = np.zeros(6, dtype=np.float64)

    projected = project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=2.1)

    assert np.allclose(projected, np.full(6, 0.35, dtype=np.float32))
    assert np.isclose(projected.sum(), 2.1, atol=1e-8)


def test_project_bounded_simplex_rejects_target_sum_just_below_minimum():
    vector = np.zeros(2, dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=0.1, upper=0.9, target_sum=0.2 - 5e-11)
    except ValueError as exc:
        assert "target_sum is outside feasible bounds" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_project_bounded_simplex_rejects_target_sum_just_above_maximum():
    vector = np.zeros(2, dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=0.1, upper=0.9, target_sum=1.8 + 5e-11)
    except ValueError as exc:
        assert "target_sum is outside feasible bounds" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_project_bounded_simplex_rejects_non_finite_scalar_parameters():
    vector = np.zeros(6, dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=np.inf)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_project_bounded_simplex_accepts_numpy_scalar_parameters():
    vector = np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15], dtype=np.float64)

    projected = project_bounded_simplex(
        vector,
        lower=np.array(0.05),
        upper=np.float64(0.35),
        target_sum=np.array(1.0),
        atol=np.float64(1e-10),
    )

    assert np.allclose(projected, vector)


def test_project_bounded_simplex_rejects_non_scalar_parameters():
    vector = np.zeros(6, dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=np.array([0.05]), upper=0.35, target_sum=1.0)
    except ValueError as exc:
        assert "scalar" in str(exc) or "finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    try:
        project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=[1.0])
    except ValueError as exc:
        assert "scalar" in str(exc) or "finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_project_bounded_simplex_rejects_negative_atol():
    vector = np.zeros(6, dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=1.0, atol=-1e-10)
    except ValueError as exc:
        assert "atol" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_project_bounded_simplex_rejects_lower_above_upper():
    vector = np.zeros(6, dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=0.35, upper=0.05, target_sum=1.0)
    except ValueError as exc:
        assert "lower must be less than or equal to upper" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_project_bounded_simplex_rejects_non_1d_input():
    matrix = np.zeros((2, 3), dtype=np.float64)

    try:
        project_bounded_simplex(matrix, lower=0.05, upper=0.35, target_sum=1.0)
    except ValueError as exc:
        assert "one-dimensional" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_project_bounded_simplex_rejects_non_finite_values():
    vector = np.array([0.2, np.nan, 0.2, 0.2, 0.1, 0.1], dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=1.0)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
