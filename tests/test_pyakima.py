"""Tests for the public pyakima spline helpers.

Copyright 2026 Matthew C. Digman
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numba import njit

from pyakima.pyakima import (
    AkimaSpline,
    SplineCoeffs,
    akima_create_helper,
    cubic_call,
    cubic_call_scalar,
    cubic_call_vector,
    cubic_call_vector_linear,
    spline_single_knot_eval,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _affine_control_points(dtype: type[np.floating] = np.float64) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(5, dtype=dtype)
    y = 2 * x + dtype(1)
    return x, y


def _nonlinear_control_points(dtype: type[np.floating] = np.float64) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    y = np.array([0.0, 1.0, 0.5, 2.0, -1.0, 3.0], dtype=dtype)
    return x, y


def _irregular_nonlinear_control_points(dtype: type[np.floating] = np.float64) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([0.0, 0.5, 2.0, 3.0, 5.0, 8.0], dtype=dtype)
    y = np.array([0.0, 0.5, 3.5, 2.5, 8.5, 10.0], dtype=dtype)
    return x, y


@pytest.fixture(name='irregular_nonlinear_control_points')
def _irregular_nonlinear_control_points_fixture() -> tuple[np.ndarray, np.ndarray]:
    return _irregular_nonlinear_control_points()


@pytest.fixture(
    name='irregular_evaluation_case',
    params=[
        pytest.param((np.array([0.0, 0.25, 0.5, 1.25, 2.6, 4.25, 6.5, 8.0]), False), id='forward-interior'),
        pytest.param((np.array([8.0, 6.5, 4.25, 2.6, 1.25, 0.5, 0.25, 0.0]), False), id='reverse-interior'),
        pytest.param((np.array([2.6, 0.25, 6.5, 1.25, 8.0, 0.5, 4.25, 0.0]), False), id='shuffled-interior'),
        pytest.param((np.array([-1.0, 0.0, 0.25, 1.25, 4.25, 8.0, 9.5]), True), id='forward-extrap'),
        pytest.param((np.array([9.5, 8.0, 4.25, 1.25, 0.25, 0.0, -1.0]), True), id='reverse-extrap'),
        pytest.param((np.array([4.25, -1.0, 8.0, 0.25, 9.5, 1.25, 0.0]), True), id='shuffled-extrap'),
    ],
)
def _irregular_evaluation_case_fixture(request: pytest.FixtureRequest) -> tuple[np.ndarray, bool]:
    xint, has_extrapolation = request.param
    return xint.copy(), has_extrapolation


def _assert_same_float_values(
    actual: np.ndarray | np.floating | float,
    expected: np.ndarray | np.floating | float,
    *,
    maxulp: int = 4,
) -> None:
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    assert actual_arr.shape == expected_arr.shape

    actual_nan = np.isnan(actual_arr)
    expected_nan = np.isnan(expected_arr)
    np.testing.assert_array_equal(actual_nan, expected_nan)

    finite = np.isfinite(actual_arr) & np.isfinite(expected_arr)
    if finite.any():
        np.testing.assert_array_max_ulp(actual_arr[finite], expected_arr[finite], maxulp=maxulp)

    nonfinite_not_nan = ~(finite | actual_nan)
    if nonfinite_not_nan.any():
        np.testing.assert_array_equal(actual_arr[nonfinite_not_nan], expected_arr[nonfinite_not_nan])


def _assert_coefficients_equal(actual: SplineCoeffs, expected: SplineCoeffs, *, maxulp: int = 4) -> None:
    assert actual.n_control == expected.n_control
    for name in ('x', 'y', 'a', 'b', 'c', 'd'):
        _assert_same_float_values(getattr(actual, name), getattr(expected, name), maxulp=maxulp)


def _single_knot_derivative(xint: float, spline: SplineCoeffs, i: int) -> float:
    dx = xint - spline.x[i]
    return float(spline.b[i] + 2.0 * spline.c[i] * dx + 3.0 * spline.d[i] * dx**2)


def _outside_interval_mask(size: int, first: int, last: int) -> np.ndarray:
    mask = np.ones(size, dtype=bool)
    mask[max(first, 0) : min(last + 1, size)] = False
    return mask


def _interval_slice(size: int, first: int, last: int) -> slice:
    return slice(max(first, 0), min(last + 1, size))


def _assert_component_unchanged_outside_interval(
    baseline: SplineCoeffs,
    actual: SplineCoeffs,
    component: str,
    first: int,
    last: int,
) -> None:
    baseline_values = getattr(baseline, component)
    actual_values = getattr(actual, component)
    mask = _outside_interval_mask(baseline_values.size, first, last)
    _assert_same_float_values(actual_values[mask], baseline_values[mask], maxulp=0)


def _power_of_two(dtype: type[np.floating], exponent: int) -> np.floating:
    return dtype(np.ldexp(1.0, exponent))


def _power_of_two_affine_points(
    dtype: type[np.floating],
    *,
    h_exponent: int,
    slope_exponent: int,
) -> tuple[np.ndarray, np.ndarray, np.floating]:
    h = _power_of_two(dtype, h_exponent)
    delta_y = _power_of_two(dtype, h_exponent + slope_exponent)
    index = np.arange(12, dtype=dtype)
    return index * h, index * delta_y, _power_of_two(dtype, slope_exponent)


def _typed_affine_spline(dtype: type[np.floating]) -> SplineCoeffs:
    x, y = _affine_control_points(dtype)
    return SplineCoeffs(
        x=x,
        y=y,
        n_control=x.size,
        a=y[:-1].copy(),
        b=np.full(x.size - 1, dtype(2), dtype=dtype),
        c=np.zeros(x.size - 1, dtype=dtype),
        d=np.zeros(x.size - 1, dtype=dtype),
    )


@njit()
def _jitted_cubic_call_scalar(xint: float, spline: SplineCoeffs, ext: int) -> float:
    return cubic_call(xint, spline, ext)


@njit()
def _jitted_cubic_call_vector(xint: np.ndarray, spline: SplineCoeffs, ext: int) -> np.ndarray:
    return cubic_call(xint, spline, ext)


@pytest.mark.parametrize('corner_model', [0, 1, 2, 'non-rounded', 'akima', 'makima'])
def test_minimum_length_affine_spline_reproduces_line_for_all_corner_models(corner_model: int | str) -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y, ext=0, corner_model=corner_model)
    xint = np.array([-1.0, 0.0, 0.25, 1.5, 3.75, 4.0, 5.0])

    np.testing.assert_array_equal(spline.spline.a, y[:-1])
    np.testing.assert_array_equal(spline.spline.b, np.full(x.size - 1, 2.0))
    np.testing.assert_array_equal(spline.spline.c, np.zeros(x.size - 1))
    np.testing.assert_array_equal(spline.spline.d, np.zeros(x.size - 1))
    np.testing.assert_array_equal(spline(xint), 2 * xint + 1)


def test_create_rejects_length_mismatch_and_nonincreasing_controls() -> None:
    x, y = _affine_control_points()

    with pytest.raises(ValueError, match='Need at least 5 control points'):
        akima_create_helper(x[:4], y[:4])

    with pytest.raises(ValueError, match='Input sizes must match'):
        akima_create_helper(x, y[:-1])

    duplicate_x = x.copy()
    duplicate_x[2] = duplicate_x[1]
    with pytest.raises(ValueError, match='x must be monotonically increasing'):
        akima_create_helper(duplicate_x, y)

    decreasing_x = x.copy()
    decreasing_x[2] = decreasing_x[1] - 1.0
    with pytest.raises(ValueError, match='x must be monotonically increasing'):
        akima_create_helper(decreasing_x, y)

    nan_x = x.copy()
    nan_x[2] = np.nan
    with pytest.raises(ValueError, match='x must be monotonically increasing'):
        akima_create_helper(nan_x, y)


@pytest.mark.parametrize(
    ('integer_model', 'string_model', 'denom_small_cut'),
    [(0, 'non-rounded', 0.0), (1, 'akima', 1.0e-9), (2, 'makima', 0.0)],
)
def test_corner_model_string_aliases_match_integer_models_when_cut_is_explicit(
    integer_model: int,
    string_model: str,
    denom_small_cut: float,
) -> None:
    x, y = _nonlinear_control_points()

    int_spline = AkimaSpline(x, y, corner_model=integer_model, denom_small_cut=denom_small_cut)
    string_spline = AkimaSpline(x, y, corner_model=string_model, denom_small_cut=denom_small_cut)

    assert string_spline.corner_model == int_spline.corner_model
    assert string_spline.denom_small_cut == int_spline.denom_small_cut
    _assert_coefficients_equal(string_spline.spline, int_spline.spline)


@pytest.mark.parametrize(
    ('integer_model', 'string_model'),
    [(0, 'non-rounded'), (1, 'akima'), (2, 'makima')],
)
def test_corner_model_string_aliases_use_same_default_denominator_cut_as_integer_models(
    integer_model: int,
    string_model: str,
) -> None:
    x, y = _nonlinear_control_points()

    int_spline = AkimaSpline(x, y, corner_model=integer_model)
    string_spline = AkimaSpline(x, y, corner_model=string_model)

    assert string_spline.corner_model == int_spline.corner_model
    assert string_spline.denom_small_cut == int_spline.denom_small_cut
    _assert_coefficients_equal(string_spline.spline, int_spline.spline)


def test_invalid_corner_model_raises_value_error() -> None:
    x, y = _affine_control_points()

    with pytest.raises(ValueError, match='Unrecognized option for corner model'):
        AkimaSpline(x, y, corner_model=99)

    with pytest.raises(ValueError, match='Unrecognized option for corner model'):
        AkimaSpline(x, y, corner_model='unknown')

    with pytest.raises(ValueError, match='Unrecognized option for corner model'):
        akima_create_helper(x, y, corner_model=99)


def test_explicit_denominator_cut_changes_corner_branch_when_large_enough() -> None:
    x = np.arange(7, dtype=np.float64)
    y = np.array([0.0, 1.0, 4.0, 2.0, 8.0, 9.0, 12.0])

    no_cut = AkimaSpline(x, y, corner_model='akima', denom_small_cut=0.0)
    large_cut = AkimaSpline(x, y, corner_model='akima', denom_small_cut=10.0)

    assert no_cut.denom_small_cut == 0.0
    assert large_cut.denom_small_cut == 10.0
    _assert_same_float_values(
        no_cut.spline.b,
        np.array([0.0, 11.0 / 7.0, 2.0, 2.0, 2.0, 17.0 / 7.0]),
    )
    _assert_same_float_values(
        large_cut.spline.b,
        np.array([0.0, 2.0, 0.5, 2.0, 3.5, 2.0]),
    )
    _assert_same_float_values(
        large_cut.spline.c,
        np.array([1.0, 4.5, -9.0, 10.5, -6.0, 1.0]),
    )
    _assert_same_float_values(
        large_cut.spline.d,
        np.array([0.0, -3.5, 6.5, -6.5, 3.5, 0.0]),
    )


@pytest.mark.parametrize('denom_small_cut', [-1.0, np.inf, -np.inf, np.nan])
def test_create_helper_rejects_negative_or_nonfinite_denominator_cut(denom_small_cut: float) -> None:
    x, y = _nonlinear_control_points()

    with pytest.raises(ValueError, match='denom_small_cut must be non-negative and finite'):
        akima_create_helper(x, y, denom_small_cut=denom_small_cut)


@pytest.mark.parametrize('denom_small_cut', [-1.0, np.inf, -np.inf])
def test_akima_spline_rejects_negative_or_nonfinite_explicit_denominator_cut(denom_small_cut: float) -> None:
    x, y = _nonlinear_control_points()

    with pytest.raises(ValueError, match='denom_small_cut must either be non-negative and finite or nan'):
        AkimaSpline(x, y, denom_small_cut=denom_small_cut)


def test_akima_spline_accepts_nan_default_and_zero_denominator_cut() -> None:
    x, y = _nonlinear_control_points()

    defaulted = AkimaSpline(x, y, denom_small_cut=np.nan)
    explicit_zero = AkimaSpline(x, y, denom_small_cut=0.0)

    assert defaulted.denom_small_cut == 0.0
    assert explicit_zero.denom_small_cut == 0.0


@pytest.mark.parametrize('linear_vector_calls', [0, 1])
def test_linear_vector_calls_keyword_does_not_change_values(linear_vector_calls: int) -> None:
    x, y = _nonlinear_control_points()
    xint = np.array([4.5, 0.0, 2.25, 1.0, -0.5, 5.5, 3.75, 0.25])

    baseline = AkimaSpline(x, y, ext=0, linear_vector_calls=0)(xint)
    actual = AkimaSpline(x, y, ext=0, linear_vector_calls=linear_vector_calls)(xint)

    _assert_same_float_values(actual, baseline, maxulp=4)


@pytest.mark.parametrize('corner_model', [0, 1, 2])
def test_integer_control_points_match_float_cast_and_yield_float_coefficients(corner_model: int) -> None:
    # integer x/y must not produce truncated integer coefficients; the result should match
    # the same data cast to float rather than silently differing (non-affine data exposes this).
    x_int = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    y_int = np.array([0, 1, 8, 27, 10, 5, 2], dtype=np.int64)

    int_spline = AkimaSpline(x_int, y_int, ext=0, corner_model=corner_model)
    float_spline = AkimaSpline(x_int.astype(np.float64), y_int.astype(np.float64), ext=0, corner_model=corner_model)

    for name in ('a', 'b', 'c', 'd'):
        assert np.issubdtype(getattr(int_spline.spline, name).dtype, np.floating)

    xint = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    _assert_same_float_values(int_spline(xint), float_spline(xint))


def test_invalid_linear_vector_calls_raises_value_error() -> None:
    x, y = _affine_control_points()

    with pytest.raises(ValueError, match='linear_vector_calls must be in'):
        AkimaSpline(x, y, linear_vector_calls=2)


def test_single_knot_eval_accepts_scalar_and_vector_inputs() -> None:
    x, y = _affine_control_points()
    spline = akima_create_helper(x, y)

    scalar = spline_single_knot_eval(np.float64(1.25), spline, 1)
    vector_x = np.array([1.0, 1.25, 1.5, 1.75])
    vector = spline_single_knot_eval(vector_x, spline, 1)

    assert scalar == 3.5
    np.testing.assert_array_equal(vector, 2 * vector_x + 1)


@pytest.mark.parametrize('corner_model', [0, 1])
def test_non_affine_basic_akima_coefficients_match_hand_computed_values(corner_model: int) -> None:
    x = np.arange(6, dtype=np.float64)
    y = np.array([0.0, 1.0, 3.0, 7.0, 14.0, 25.0])

    spline = akima_create_helper(x, y, corner_model=corner_model, denom_small_cut=0.0)

    np.testing.assert_array_equal(spline.a, y[:-1])
    _assert_same_float_values(spline.b, np.array([1.0 / 2.0, 4.0 / 3.0, 5.0 / 2.0, 5.0, 61.0 / 7.0]), maxulp=18)
    _assert_same_float_values(spline.c, np.array([2.0 / 3.0, 5.0 / 6.0, 2.0, 16.0 / 7.0, 18.0 / 7.0]), maxulp=18)
    _assert_same_float_values(
        spline.d,
        np.array([-1.0 / 6.0, -1.0 / 6.0, -1.0 / 2.0, -2.0 / 7.0, -2.0 / 7.0]),
        maxulp=18,
    )


def test_non_affine_makima_coefficients_match_hand_computed_values() -> None:
    x = np.arange(6, dtype=np.float64)
    y = np.array([0.0, 1.0, 3.0, 7.0, 14.0, 25.0])

    spline = akima_create_helper(x, y, corner_model=2, denom_small_cut=0.0)

    np.testing.assert_array_equal(spline.a, y[:-1])
    _assert_same_float_values(
        spline.b,
        np.array([3.0 / 8.0, 16.0 / 13.0, 27.0 / 11.0, 29.0 / 6.0, 25.0 / 3.0]),
        maxulp=11,
    )
    _assert_same_float_values(
        spline.c,
        np.array([53.0 / 52.0, 155.0 / 143.0, 149.0 / 66.0, 3.0, 194.0 / 51.0]),
        maxulp=11,
    )
    _assert_same_float_values(
        spline.d,
        np.array([-41.0 / 104.0, -45.0 / 143.0, -47.0 / 66.0, -5.0 / 6.0, -58.0 / 51.0]),
        maxulp=11,
    )


@pytest.mark.parametrize('corner_model', [0, 1, 2])
def test_subsplines_interpolate_from_left_and_are_c1_continuous_without_sharp_corners(corner_model: int) -> None:
    x = np.arange(6, dtype=np.float64)
    y = np.array([0.0, 1.0, 3.0, 7.0, 14.0, 25.0])
    spline = akima_create_helper(x, y, corner_model=corner_model, denom_small_cut=0.0)

    left_endpoint_values = np.array([spline_single_knot_eval(x[i + 1], spline, i) for i in range(x.size - 1)])
    _assert_same_float_values(left_endpoint_values, y[1:], maxulp=4)

    left_derivatives = np.array([_single_knot_derivative(float(x[i + 1]), spline, i) for i in range(x.size - 2)])
    right_derivatives = spline.b[1:]
    _assert_same_float_values(left_derivatives, right_derivatives, maxulp=4)


@pytest.mark.parametrize(
    ('corner_model', 'denom_small_cut', 'expected_b', 'expected_c', 'expected_d'),
    [
        (
            0,
            0.0,
            np.array([1.0, 1.0, 3.0, 3.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0, -1.0]),
        ),
        (
            1,
            0.0,
            np.array([1.0, 1.0, 2.0, 3.0, 3.0]),
            np.array([0.0, -1.0, 2.0, 0.0, 3.0]),
            np.array([0.0, 1.0, -1.0, 0.0, -1.0]),
        ),
        (
            2,
            10.0,
            np.array([1.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, -1.0, 2.0, -1.0, 1.0]),
            np.array([0.0, 1.0, -1.0, 1.0, 0.0]),
        ),
    ],
)
def test_corner_branch_coefficients_match_sharp_and_rounded_models(
    corner_model: int,
    denom_small_cut: float,
    expected_b: np.ndarray,
    expected_c: np.ndarray,
    expected_d: np.ndarray,
) -> None:
    x = np.arange(6, dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 5.0, 8.0, 13.0])

    spline = akima_create_helper(x, y, corner_model=corner_model, denom_small_cut=denom_small_cut)

    _assert_same_float_values(spline.b, expected_b)
    _assert_same_float_values(spline.c, expected_c)
    _assert_same_float_values(spline.d, expected_d)


def test_heaviside_step_default_corner_models_agree_on_monotone_transition() -> None:
    x = np.arange(7, dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    midpoints = x[:-1] + 0.5

    nonrounded = AkimaSpline(x, y, ext=0, corner_model=0)
    akima = AkimaSpline(x, y, ext=0, corner_model=1)
    makima = AkimaSpline(x, y, ext=0, corner_model=2)

    _assert_same_float_values(nonrounded.spline.b, np.zeros(6))
    _assert_same_float_values(nonrounded.spline.c, np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]))
    _assert_same_float_values(nonrounded.spline.d, np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0]))
    _assert_same_float_values(nonrounded(midpoints), np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0]))
    _assert_coefficients_equal(akima.spline, nonrounded.spline)
    _assert_coefficients_equal(makima.spline, nonrounded.spline)


def test_heaviside_step_corner_cut_distinguishes_sharp_and_rounded_models() -> None:
    x = np.arange(7, dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    midpoints = x[:-1] + 0.5

    sharp = AkimaSpline(x, y, ext=0, corner_model=0, denom_small_cut=2.0)
    rounded = AkimaSpline(x, y, ext=0, corner_model=1, denom_small_cut=2.0)
    makima = AkimaSpline(x, y, ext=0, corner_model=2, denom_small_cut=2.0)

    _assert_same_float_values(sharp.spline.b, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    _assert_same_float_values(sharp.spline.c, np.zeros(6))
    _assert_same_float_values(sharp.spline.d, np.zeros(6))
    _assert_same_float_values(sharp(midpoints), np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0]))

    expected_rounded_b = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
    expected_rounded_c = np.array([0.0, -0.5, 1.5, -1.0, 0.0, 0.0])
    expected_rounded_d = np.array([0.0, 0.5, -1.0, 0.5, 0.0, 0.0])
    expected_rounded_midpoints = np.array([0.0, -1.0 / 16.0, 0.5, 17.0 / 16.0, 1.0, 1.0])

    _assert_same_float_values(rounded.spline.b, expected_rounded_b)
    _assert_same_float_values(rounded.spline.c, expected_rounded_c)
    _assert_same_float_values(rounded.spline.d, expected_rounded_d)
    _assert_same_float_values(rounded(midpoints), expected_rounded_midpoints)
    _assert_coefficients_equal(makima.spline, rounded.spline)
    _assert_same_float_values(makima(midpoints), expected_rounded_midpoints)


def test_abs_like_slope_sign_change_distinguishes_all_corner_models() -> None:
    x = np.arange(-3.0, 4.0, dtype=np.float64)
    y = np.where(x < 0.0, -2.0 * x, x)
    near_corner = np.array([-0.5, 0.5])

    sharp = AkimaSpline(x, y, ext=0, corner_model=0, denom_small_cut=0.0)
    rounded = AkimaSpline(x, y, ext=0, corner_model=1, denom_small_cut=0.0)
    makima = AkimaSpline(x, y, ext=0, corner_model=2, denom_small_cut=0.0)

    _assert_same_float_values(sharp.spline.b, np.array([-2.0, -2.0, -2.0, 1.0, 1.0, 1.0]))
    _assert_same_float_values(sharp.spline.c, np.zeros(6))
    _assert_same_float_values(sharp.spline.d, np.zeros(6))
    _assert_same_float_values(sharp(near_corner), np.array([1.0, 0.5]))

    _assert_same_float_values(rounded.spline.b, np.array([-2.0, -2.0, -2.0, -0.5, 1.0, 1.0]))
    _assert_same_float_values(rounded.spline.c, np.array([0.0, 0.0, -1.5, 3.0, 0.0, 0.0]))
    _assert_same_float_values(rounded.spline.d, np.array([0.0, 0.0, 1.5, -1.5, 0.0, 0.0]))
    _assert_same_float_values(rounded(near_corner), np.array([13.0 / 16.0, 5.0 / 16.0]))

    _assert_same_float_values(makima.spline.b, np.array([-2.0, -2.0, -2.0, 0.0, 1.0, 1.0]))
    _assert_same_float_values(makima.spline.c, np.array([0.0, 0.0, -2.0, 2.0, 0.0, 0.0]))
    _assert_same_float_values(makima.spline.d, np.array([0.0, 0.0, 2.0, -1.0, 0.0, 0.0]))
    _assert_same_float_values(makima(near_corner), np.array([3.0 / 4.0, 3.0 / 8.0]))

    left_interval = 2
    right_interval = 3
    corner_x = 0.0
    _assert_same_float_values(
        np.array(
            [
                _single_knot_derivative(corner_x, sharp.spline, left_interval),
                sharp.spline.b[right_interval],
            ]
        ),
        np.array([-2.0, 1.0]),
    )
    _assert_same_float_values(
        np.array(
            [
                _single_knot_derivative(corner_x, rounded.spline, left_interval),
                rounded.spline.b[right_interval],
            ]
        ),
        np.array([-0.5, -0.5]),
    )
    _assert_same_float_values(
        np.array(
            [
                _single_knot_derivative(corner_x, makima.spline, left_interval),
                makima.spline.b[right_interval],
            ]
        ),
        np.array([0.0, 0.0]),
    )


def test_dirac_delta_sample_default_corner_models_agree_on_symmetric_pulse() -> None:
    x = np.arange(7, dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    midpoints = x[:-1] + 0.5

    nonrounded = AkimaSpline(x, y, ext=0, corner_model=0)
    akima = AkimaSpline(x, y, ext=0, corner_model=1)
    makima = AkimaSpline(x, y, ext=0, corner_model=2)

    _assert_same_float_values(nonrounded.spline.b, np.zeros(6))
    _assert_same_float_values(nonrounded.spline.c, np.array([0.0, 0.0, 3.0, -3.0, 0.0, 0.0]))
    _assert_same_float_values(nonrounded.spline.d, np.array([0.0, 0.0, -2.0, 2.0, 0.0, 0.0]))
    _assert_same_float_values(nonrounded(midpoints), np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0]))
    _assert_coefficients_equal(akima.spline, nonrounded.spline)
    _assert_coefficients_equal(makima.spline, nonrounded.spline)


def test_dirac_delta_sample_corner_cut_distinguishes_sharp_and_rounded_models() -> None:
    x = np.arange(7, dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    midpoints = x[:-1] + 0.5

    sharp = AkimaSpline(x, y, ext=0, corner_model=0, denom_small_cut=2.0)
    rounded = AkimaSpline(x, y, ext=0, corner_model=1, denom_small_cut=2.0)
    makima = AkimaSpline(x, y, ext=0, corner_model=2, denom_small_cut=2.0)

    _assert_same_float_values(sharp.spline.b, np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0]))
    _assert_same_float_values(sharp.spline.c, np.zeros(6))
    _assert_same_float_values(sharp.spline.d, np.zeros(6))
    _assert_same_float_values(sharp(midpoints), np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0]))

    expected_rounded_b = np.array([0.0, 0.0, 0.5, 0.0, -0.5, 0.0])
    expected_rounded_c = np.array([0.0, -0.5, 2.0, -2.5, 1.0, 0.0])
    expected_rounded_d = np.array([0.0, 0.5, -1.5, 1.5, -0.5, 0.0])
    expected_rounded_midpoints = np.array([0.0, -1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0, 0.0])

    _assert_same_float_values(rounded.spline.b, expected_rounded_b)
    _assert_same_float_values(rounded.spline.c, expected_rounded_c)
    _assert_same_float_values(rounded.spline.d, expected_rounded_d)
    _assert_same_float_values(rounded(midpoints), expected_rounded_midpoints)
    _assert_coefficients_equal(makima.spline, rounded.spline)
    _assert_same_float_values(makima(midpoints), expected_rounded_midpoints)


def test_reachable_zero_left_weight_path_matches_hand_computed_coefficients() -> None:
    x = np.arange(6, dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 17.0])

    spline = akima_create_helper(x, y, corner_model=0, denom_small_cut=0.0)

    _assert_same_float_values(spline.b, np.array([1.0, 1.0, 1.0, 4.0, 6.0]))
    _assert_same_float_values(spline.c, np.array([0.0, 0.0, 3.0, 1.0, 1.0]))
    _assert_same_float_values(spline.d, np.array([0.0, 0.0, -1.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    ('corner_model', 'expected_b', 'expected_c', 'expected_d'),
    [
        (
            0,
            np.array([1.0 / 2.0, 5.0 / 4.0, 7.0 / 5.0, 13.0 / 11.0, 19.0 / 13.0]),
            np.array([3.0 / 2.0, 7.0 / 5.0, -384.0 / 55.0, 370.0 / 143.0, -35.0 / 156.0]),
            np.array([-1.0, -3.0 / 5.0, 252.0 / 55.0, -120.0 / 143.0, -5.0 / 156.0]),
        ),
        (
            1,
            np.array([1.0 / 2.0, 5.0 / 4.0, 7.0 / 5.0, 13.0 / 11.0, 19.0 / 13.0]),
            np.array([3.0 / 2.0, 7.0 / 5.0, -384.0 / 55.0, 370.0 / 143.0, -35.0 / 156.0]),
            np.array([-1.0, -3.0 / 5.0, 252.0 / 55.0, -120.0 / 143.0, -5.0 / 156.0]),
        ),
        (
            2,
            np.array([3.0 / 8.0, 13.0 / 10.0, 1.0, 25.0 / 31.0, 49.0 / 33.0]),
            np.array([19.0 / 10.0, 8.0 / 5.0, -180.0 / 31.0, 3019.0 / 1023.0, -479.0 / 1584.0]),
            np.array([-13.0 / 10.0, -34.0 / 45.0, 118.0 / 31.0, -1897.0 / 2046.0, -41.0 / 4752.0]),
        ),
    ],
)
def test_irregular_grid_coefficients_match_hand_computed_values(
    corner_model: int,
    expected_b: np.ndarray,
    expected_c: np.ndarray,
    expected_d: np.ndarray,
) -> None:
    x, y = _irregular_nonlinear_control_points()
    assert not np.all(np.diff(x) == np.diff(x)[0])

    spline = akima_create_helper(x, y, corner_model=corner_model, denom_small_cut=0.0)

    np.testing.assert_array_equal(spline.a, y[:-1])
    _assert_same_float_values(spline.b, expected_b, maxulp=32)
    _assert_same_float_values(spline.c, expected_c, maxulp=32)
    _assert_same_float_values(spline.d, expected_d, maxulp=32)


@pytest.mark.parametrize('corner_model', [0, 1, 2])
def test_evaluating_exactly_at_knots_returns_control_values(corner_model: int) -> None:
    x, y = _nonlinear_control_points()
    spline = AkimaSpline(x, y, ext=0, corner_model=corner_model)

    scalar_values = np.array([cubic_call_scalar(float(knot), spline.spline, 0) for knot in x])
    vector_values = cubic_call_vector(x, spline.spline, 0)
    linear_vector_values = cubic_call_vector_linear(x, spline.spline, 0)

    _assert_same_float_values(scalar_values, y, maxulp=4)
    _assert_same_float_values(vector_values, y, maxulp=4)
    _assert_same_float_values(linear_vector_values, y, maxulp=4)
    _assert_same_float_values(spline(x), y, maxulp=4)


@pytest.mark.parametrize('corner_model', [0, 1, 2])
@pytest.mark.parametrize('ext', [0, 3])
def test_irregular_grid_evaluating_exactly_at_knots_returns_control_values(
    irregular_nonlinear_control_points: tuple[np.ndarray, np.ndarray],
    corner_model: int,
    ext: int,
) -> None:
    x, y = irregular_nonlinear_control_points
    spline = AkimaSpline(x, y, ext=ext, corner_model=corner_model)

    scalar_values = np.array([cubic_call_scalar(float(knot), spline.spline, ext) for knot in x])
    vector_values = cubic_call_vector(x, spline.spline, ext)
    linear_vector_values = cubic_call_vector_linear(x, spline.spline, ext)

    _assert_same_float_values(scalar_values, y, maxulp=4)
    _assert_same_float_values(vector_values, y, maxulp=4)
    _assert_same_float_values(linear_vector_values, y, maxulp=4)
    _assert_same_float_values(spline(x), y, maxulp=4)


def test_last_knot_evaluation_returns_control_value_exactly_with_large_coefficients() -> None:
    scale = np.ldexp(1.0, 1000)
    x = np.arange(5, dtype=np.float64)
    y = scale * np.array([0.0, 1.0, -1.0, 2.0, -3.0])
    x_last = float(x[-1])
    x_last_vector = np.array([x_last])
    expected = np.array([y[-1]])

    spline = AkimaSpline(x, y, ext=4)
    linear_spline = AkimaSpline(x, y, ext=4, linear_vector_calls=1)

    _assert_same_float_values(cubic_call_scalar(x_last, spline.spline, 4), y[-1], maxulp=0)
    _assert_same_float_values(cubic_call_vector(x_last_vector, spline.spline, 4), expected, maxulp=0)
    _assert_same_float_values(cubic_call_vector_linear(x_last_vector, spline.spline, 4), expected, maxulp=0)
    _assert_same_float_values(cubic_call(x_last, spline.spline, 4), y[-1], maxulp=0)
    _assert_same_float_values(cubic_call(x_last_vector, spline.spline, 4), expected, maxulp=0)
    _assert_same_float_values(spline(x_last), y[-1], maxulp=0)
    _assert_same_float_values(spline(x_last_vector), expected, maxulp=0)
    _assert_same_float_values(linear_spline(x_last_vector), expected, maxulp=0)


@pytest.mark.parametrize(
    ('ext', 'expected'),
    [
        (0, np.array([-1.0, 1.0, 2.0, 9.0, 11.0])),
        (1, np.array([0.0, 1.0, 2.0, 9.0, 0.0])),
        (3, np.array([1.0, 1.0, 2.0, 9.0, 9.0])),
        (4, np.array([np.nan, 1.0, 2.0, 9.0, np.nan])),
    ],
)
def test_extrapolation_modes_for_scalar_vector_and_class_calls(ext: int, expected: np.ndarray) -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y, ext=ext)
    xint = np.array([-1.0, 0.0, 0.5, 4.0, 5.0])

    scalar_values = np.array([cubic_call_scalar(float(point), spline.spline, ext) for point in xint])
    dispatch_values = cubic_call(xint, spline.spline, ext)
    vector_values = cubic_call_vector(xint, spline.spline, ext)
    linear_vector_values = cubic_call_vector_linear(xint, spline.spline, ext)
    class_values = spline(xint)

    _assert_same_float_values(scalar_values, expected)
    _assert_same_float_values(dispatch_values, expected)
    _assert_same_float_values(vector_values, expected)
    _assert_same_float_values(linear_vector_values, expected)
    _assert_same_float_values(class_values, expected)


@pytest.mark.parametrize(
    ('ext', 'expected'),
    [
        (1, np.array([6.0, 0.0, 0.0, 2.0])),
        (3, np.array([6.0, 1.0, 9.0, 2.0])),
        (4, np.array([6.0, np.nan, np.nan, 2.0])),
    ],
)
def test_vector_loop_handles_below_and_above_range_after_initial_point(ext: int, expected: np.ndarray) -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y, ext=ext)
    xint = np.array([2.5, -1.0, 5.0, 0.5])

    _assert_same_float_values(cubic_call_vector(xint, spline.spline, ext), expected, maxulp=0)
    _assert_same_float_values(spline(xint), expected, maxulp=0)


def test_vector_forward_search_loop_exhaustion_preserves_nan_result() -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y, ext=3)
    # A leading nan reaches the forward search loop; every knot comparison is false.
    xint = np.array([np.nan, 0.5])
    expected = np.array([np.nan, 2.0])

    _assert_same_float_values(cubic_call_vector(xint, spline.spline, 3), expected, maxulp=0)
    _assert_same_float_values(cubic_call(xint, spline.spline, 3), expected, maxulp=0)
    _assert_same_float_values(spline(xint), expected, maxulp=0)


def test_invalid_extrapolation_mode_raises_for_all_call_paths() -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y)
    xint = np.array([0.5, 1.5])

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        cubic_call_scalar(0.5, spline.spline, 2)

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        cubic_call_vector(xint, spline.spline, 2)

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        cubic_call_vector_linear(xint, spline.spline, 2)

    bad_ext_spline = AkimaSpline(x, y, ext=2)
    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        bad_ext_spline(0.5)

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        bad_ext_spline(xint)


@pytest.mark.parametrize(
    'xint',
    [
        np.array([-0.5, 0.0, 0.25, 1.75, 3.5, 5.25]),
        np.array([5.25, 3.5, 1.75, 0.25, 0.0, -0.5]),
        np.array([2.2, -0.5, 5.25, 1.1, 1.1, 4.0, 0.0]),
        np.array([2.2]),
        np.array([], dtype=np.float64),
    ],
)
@pytest.mark.parametrize('ext', [0, 1, 3, 4])
def test_scalar_vector_dispatch_and_linear_vector_paths_agree(xint: np.ndarray, ext: int) -> None:
    x, y = _nonlinear_control_points()
    spline = AkimaSpline(x, y, ext=ext)
    assert np.any(spline.spline.c != 0.0) or np.any(spline.spline.d != 0.0)

    scalar_values = np.array([cubic_call_scalar(float(point), spline.spline, ext) for point in xint])
    vector_values = cubic_call_vector(xint, spline.spline, ext)
    linear_vector_values = cubic_call_vector_linear(xint, spline.spline, ext)
    dispatch_values = cubic_call(xint, spline.spline, ext)
    class_values = spline(xint)
    linear_class_values = AkimaSpline(x, y, ext=ext, linear_vector_calls=1)(xint)

    _assert_same_float_values(vector_values, scalar_values, maxulp=0)
    _assert_same_float_values(linear_vector_values, scalar_values, maxulp=0)
    _assert_same_float_values(dispatch_values, scalar_values, maxulp=0)
    _assert_same_float_values(class_values, scalar_values, maxulp=0)
    _assert_same_float_values(linear_class_values, scalar_values, maxulp=0)


@pytest.mark.parametrize('ext', [0, 1, 3, 4])
def test_irregular_grid_scalar_vector_dispatch_and_linear_vector_paths_agree(
    irregular_nonlinear_control_points: tuple[np.ndarray, np.ndarray],
    irregular_evaluation_case: tuple[np.ndarray, bool],
    ext: int,
) -> None:
    x, y = irregular_nonlinear_control_points
    xint, has_extrapolation = irregular_evaluation_case
    spline = AkimaSpline(x, y, ext=ext)
    assert not np.all(np.diff(x) == np.diff(x)[0])
    assert np.any(spline.spline.c != 0.0) or np.any(spline.spline.d != 0.0)

    below_domain = xint < x[0]
    above_domain = xint > x[-1]
    assert bool(below_domain.any()) is has_extrapolation
    assert bool(above_domain.any()) is has_extrapolation

    scalar_values = np.array([cubic_call_scalar(float(point), spline.spline, ext) for point in xint])
    vector_values = cubic_call_vector(xint, spline.spline, ext)
    linear_vector_values = cubic_call_vector_linear(xint, spline.spline, ext)
    dispatch_values = cubic_call(xint, spline.spline, ext)
    class_values = spline(xint)
    linear_class_values = AkimaSpline(x, y, ext=ext, linear_vector_calls=1)(xint)

    _assert_same_float_values(vector_values, scalar_values, maxulp=0)
    _assert_same_float_values(linear_vector_values, scalar_values, maxulp=0)
    _assert_same_float_values(dispatch_values, scalar_values, maxulp=0)
    _assert_same_float_values(class_values, scalar_values, maxulp=0)
    _assert_same_float_values(linear_class_values, scalar_values, maxulp=0)

    if has_extrapolation and ext == 1:
        _assert_same_float_values(scalar_values[below_domain | above_domain], np.zeros(2), maxulp=0)
    elif has_extrapolation and ext == 3:
        _assert_same_float_values(scalar_values[below_domain], np.array([y[0]]), maxulp=0)
        _assert_same_float_values(scalar_values[above_domain], np.array([y[-1]]), maxulp=0)
    elif has_extrapolation and ext == 4:
        assert np.all(np.isnan(scalar_values[below_domain | above_domain]))


@pytest.mark.parametrize('ext', [0, 1, 3, 4])
def test_cubic_call_matches_numba_overload_for_scalar_and_vector_inputs(ext: int) -> None:
    x, y = _nonlinear_control_points()
    spline = AkimaSpline(x, y, ext=ext)
    scalar_x = 2.25
    vector_x = np.array([-0.5, 0.0, 0.25, 1.75, 3.5, 5.25])

    _assert_same_float_values(
        _jitted_cubic_call_scalar(scalar_x, spline.spline, ext),
        cubic_call(scalar_x, spline.spline, ext),
        maxulp=0,
    )
    _assert_same_float_values(
        _jitted_cubic_call_vector(vector_x, spline.spline, ext),
        cubic_call(vector_x, spline.spline, ext),
        maxulp=0,
    )


def test_cubic_call_invalid_ext_matches_numba_overload_for_scalar_and_vector_inputs() -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y)
    vector_x = np.array([0.5, 1.5])

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        cubic_call(0.5, spline.spline, 2)

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        _jitted_cubic_call_scalar(0.5, spline.spline, 2)

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        cubic_call(vector_x, spline.spline, 2)

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        _jitted_cubic_call_vector(vector_x, spline.spline, 2)


def test_cubic_call_rejects_unsupported_input_types_outside_numba() -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y)

    with pytest.raises(TypeError):
        cubic_call(1, spline.spline, 3)

    with pytest.raises(TypeError):
        cubic_call([0.5, 1.5], spline.spline, 3)  # type: ignore[call-overload]

    with pytest.raises(TypeError):
        cubic_call(0.5, spline.spline, 3.0)  # type: ignore[call-overload]

    with pytest.raises(TypeError):
        cubic_call(0.5, 1.0, 3)  # type: ignore[call-overload]

    with pytest.raises(TypeError):
        spline(1)


def test_numba_overload_rejects_non_integer_ext_type() -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y)

    @njit()
    def call_with_float_ext(xint: float, spline_coeffs: SplineCoeffs) -> float:
        return cubic_call(xint, spline_coeffs, 3.0)  # type: ignore[call-overload, no-any-return]

    with pytest.raises(TypeError, match='Unsuported type of input'):
        call_with_float_ext(0.5, spline.spline)


def test_numba_overload_rejects_non_spline_type() -> None:
    @njit()
    def call_with_float_spline(xint: float) -> float:
        return cubic_call(xint, 1.0, 3)  # type: ignore[call-overload, no-any-return]

    with pytest.raises(TypeError, match='Unsuported type of input'):
        call_with_float_spline(0.5)


def test_numba_overload_rejects_unsupported_xint_type() -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y)

    @njit()
    def call_with_integer_xint(spline_coeffs: SplineCoeffs) -> float:
        return cubic_call(1, spline_coeffs, 3)

    with pytest.raises(TypeError, match='Unsuported type of input'):
        call_with_integer_xint(spline.spline)


@pytest.mark.parametrize(
    ('ext', 'expected'),
    [
        (0, np.array([np.nan, np.nan, np.nan])),
        (1, np.array([0.0, np.nan, 0.0])),
        (3, np.array([1.0, np.nan, 9.0])),
        (4, np.array([np.nan, np.nan, np.nan])),
    ],
)
def test_nonfinite_evaluation_points(ext: int, expected: np.ndarray) -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y, ext=ext)
    xint = np.array([-np.inf, np.nan, np.inf])

    scalar_values = np.array([cubic_call_scalar(float(point), spline.spline, ext) for point in xint])
    vector_values = cubic_call_vector(xint, spline.spline, ext)
    linear_vector_values = cubic_call_vector_linear(xint, spline.spline, ext)
    class_values = spline(xint)

    _assert_same_float_values(scalar_values, expected)
    _assert_same_float_values(vector_values, expected)
    _assert_same_float_values(linear_vector_values, expected)
    _assert_same_float_values(class_values, expected)


def test_nonfinite_y_contaminates_local_region_but_leaves_distant_finite_values_usable() -> None:
    x = np.arange(10, dtype=np.float64)
    clean_y = 2 * x + 1
    y_with_nan = clean_y.copy()
    y_with_nan[5] = np.nan
    spline = AkimaSpline(x, y_with_nan, ext=0)

    xint = np.array([0.5, 1.5, 2.5, 7.5, 8.5])
    actual = spline(xint)

    _assert_same_float_values(actual[[0, 1, 4]], (2 * xint + 1)[[0, 1, 4]])
    assert np.isnan(actual[2])
    assert np.isnan(actual[3])


def test_integer_control_arrays_are_accepted_without_integer_output_dtype_guarantee() -> None:
    x = np.arange(5, dtype=np.int64)
    y = 2 * x + 1
    xint = np.array([0.0, 0.5, 2.5, 4.0])

    helper_spline = akima_create_helper(x, y)  # type: ignore[arg-type]
    object_spline = AkimaSpline(x, y, ext=0)

    np.testing.assert_array_equal(helper_spline.b, np.full(x.size - 1, 2.0))
    np.testing.assert_array_equal(helper_spline.c, np.zeros(x.size - 1))
    np.testing.assert_array_equal(helper_spline.d, np.zeros(x.size - 1))
    np.testing.assert_array_equal(object_spline(xint), 2 * xint + 1)


@pytest.mark.parametrize(
    ('x_dtype', 'y_dtype'),
    [(np.float32, np.float64), (np.float64, np.float32)],
)
def test_mixed_float_control_dtypes_are_accepted_and_coefficients_follow_x_dtype(
    x_dtype: type[np.floating],
    y_dtype: type[np.floating],
) -> None:
    x = np.arange(6, dtype=x_dtype)
    y = np.array([0.0, 1.0, 0.5, 2.0, -1.0, 3.0], dtype=y_dtype)

    helper_spline = akima_create_helper(x, y)
    object_spline = AkimaSpline(x, y, ext=0).spline

    for spline in (helper_spline, object_spline):
        assert spline.x.dtype == np.dtype(x_dtype)
        assert spline.y.dtype == np.dtype(y_dtype)
        assert {spline.a.dtype, spline.b.dtype, spline.c.dtype, spline.d.dtype} == {np.dtype(x_dtype)}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('corner_model', [0, 1, 2])
def test_large_dynamic_range_control_point_has_strict_local_coefficient_kernel(
    dtype: type[np.floating],
    corner_model: int,
) -> None:
    x = np.arange(17, dtype=dtype)
    y = np.array(
        [
            0.0,
            0.25,
            -0.5,
            0.125,
            0.75,
            -0.25,
            0.375,
            -0.625,
            0.5,
            -0.125,
            0.875,
            0.0,
            -0.375,
            0.625,
            -0.75,
            0.25,
            -0.125,
        ],
        dtype=dtype,
    )
    changed = y.copy()
    changed_index = 8
    changed[changed_index] = _power_of_two(dtype, 90 if dtype is np.float32 else 900)

    baseline = akima_create_helper(x, y, corner_model=corner_model)
    large_dynamic_range = akima_create_helper(x, changed, corner_model=corner_model)

    _assert_component_unchanged_outside_interval(baseline, large_dynamic_range, 'a', changed_index, changed_index)
    _assert_component_unchanged_outside_interval(
        baseline,
        large_dynamic_range,
        'b',
        changed_index - 2,
        changed_index + 2,
    )
    for component in ('c', 'd'):
        _assert_component_unchanged_outside_interval(
            baseline,
            large_dynamic_range,
            component,
            changed_index - 3,
            changed_index + 2,
        )

    assert large_dynamic_range.a[changed_index] != baseline.a[changed_index]
    b_kernel = _interval_slice(large_dynamic_range.b.size, changed_index - 2, changed_index + 2)
    cd_kernel = _interval_slice(large_dynamic_range.c.size, changed_index - 3, changed_index + 2)
    assert not np.array_equal(large_dynamic_range.b[b_kernel], baseline.b[b_kernel])
    assert not np.array_equal(large_dynamic_range.c[cd_kernel], baseline.c[cd_kernel])
    assert not np.array_equal(large_dynamic_range.d[cd_kernel], baseline.d[cd_kernel])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_overflowing_control_value_differences_are_confined_to_strict_coefficient_kernel(
    dtype: type[np.floating],
) -> None:
    x = np.arange(16, dtype=dtype)
    y = (np.sin(np.arange(16)) * 0.125).astype(dtype)
    baseline = akima_create_helper(x, y)

    changed = y.copy()
    changed_index = 7
    changed[changed_index] = np.finfo(dtype).max  # pylint: disable=no-member
    changed[changed_index + 1] = -np.finfo(dtype).max  # pylint: disable=no-member
    overflowed = akima_create_helper(x, changed)

    _assert_component_unchanged_outside_interval(baseline, overflowed, 'a', changed_index, changed_index + 1)
    _assert_component_unchanged_outside_interval(baseline, overflowed, 'b', changed_index - 2, changed_index + 3)
    for component in ('c', 'd'):
        _assert_component_unchanged_outside_interval(
            baseline,
            overflowed,
            component,
            changed_index - 3,
            changed_index + 3,
        )

    b_kernel = _interval_slice(overflowed.b.size, changed_index - 2, changed_index + 3)
    cd_kernel = _interval_slice(overflowed.c.size, changed_index - 3, changed_index + 3)
    assert np.any(~np.isfinite(overflowed.b[b_kernel]))
    assert np.any(~np.isfinite(overflowed.c[cd_kernel]))
    assert np.any(~np.isfinite(overflowed.d[cd_kernel]))


def test_finite_control_values_with_nonfinite_differences_produce_local_nan_coefficients() -> None:
    x = np.arange(8, dtype=np.float64)
    finite_max = np.finfo(np.float64).max  # pylint: disable=no-member
    y = np.array([0.0, finite_max, -finite_max, 0.0, 1.0, 2.0, 3.0, 4.0])

    spline = akima_create_helper(x, y)

    assert np.all(np.isfinite(spline.y))
    with np.errstate(over='ignore'):
        finite_diff_mask = np.isfinite(np.diff(spline.y))
    np.testing.assert_array_equal(finite_diff_mask, np.array([True, False, True, True, True, True, True]))
    np.testing.assert_array_equal(np.isnan(spline.b), np.array([True, True, True, True, False, False, False]))
    np.testing.assert_array_equal(np.isnan(spline.c), np.array([True, True, True, True, False, False, False]))
    np.testing.assert_array_equal(np.isnan(spline.d), np.array([True, True, True, True, False, False, False]))
    np.testing.assert_array_equal(spline.b[4:], np.ones(3))
    np.testing.assert_array_equal(spline.c[4:], np.zeros(3))
    np.testing.assert_array_equal(spline.d[4:], np.zeros(3))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_subnormal_x_spacing_division_overflow_returns_nonfinite_local_coefficients(
    dtype: type[np.floating],
) -> None:
    step = np.nextafter(dtype(0), dtype(1), dtype=dtype)
    x = np.array([-3.0, -2.0, -1.0, 0.0, step, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    y = np.array([0.0, 0.5, -0.25, 0.0, 1.0, 0.25, -0.5, 0.75, 0.1, -0.2], dtype=dtype)

    clean_x = x.copy()
    clean_x[4] = dtype(0.5)
    baseline = akima_create_helper(clean_x, y)
    tiny_interval = akima_create_helper(x, y)

    _assert_component_unchanged_outside_interval(baseline, tiny_interval, 'a', 0, -1)
    _assert_component_unchanged_outside_interval(baseline, tiny_interval, 'b', 2, 6)
    for component in ('c', 'd'):
        _assert_component_unchanged_outside_interval(baseline, tiny_interval, component, 1, 6)

    b_kernel = _interval_slice(tiny_interval.b.size, 2, 6)
    cd_kernel = _interval_slice(tiny_interval.c.size, 1, 6)
    assert np.any(~np.isfinite(tiny_interval.b[b_kernel]))
    assert np.any(~np.isfinite(tiny_interval.c[cd_kernel]))
    assert np.any(~np.isfinite(tiny_interval.d[cd_kernel]))


@pytest.mark.parametrize(
    ('dtype', 'h_exponent', 'slope_exponent'),
    [(np.float32, -50, 125), (np.float64, -500, 1021)],
)
def test_near_overflow_affine_spline_keeps_zero_higher_order_coefficients(
    dtype: type[np.floating],
    h_exponent: int,
    slope_exponent: int,
) -> None:
    x, y, slope = _power_of_two_affine_points(dtype, h_exponent=h_exponent, slope_exponent=slope_exponent)

    spline = akima_create_helper(x, y)

    expected_slope = np.full(spline.b.shape, float(slope), dtype=spline.b.dtype)
    _assert_same_float_values(spline.b, expected_slope, maxulp=4)
    np.testing.assert_array_equal(spline.c, np.zeros_like(spline.c))
    np.testing.assert_array_equal(spline.d, np.zeros_like(spline.d))


@pytest.mark.parametrize(
    ('dtype', 'h_exponent', 'slope_exponent'),
    [(np.float32, -50, 128), (np.float64, -500, 1023)],
)
def test_actual_multiplier_overflow_in_affine_spline_produces_ieee_nonfinite_coefficients(
    dtype: type[np.floating],
    h_exponent: int,
    slope_exponent: int,
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='overflow encountered in cast')
        x, y, _ = _power_of_two_affine_points(dtype, h_exponent=h_exponent, slope_exponent=slope_exponent)

    spline = akima_create_helper(x, y)
    assert np.all(np.isfinite(spline.a))
    assert np.any(~np.isfinite(spline.b))
    assert np.any(~np.isfinite(spline.c))
    assert np.any(~np.isfinite(spline.d))


@pytest.mark.parametrize(
    ('b', 'c', 'd'),
    [
        (np.inf, 0.0, 0.0),
        (0.0, np.inf, 0.0),
        (0.0, 0.0, np.inf),
    ],
)
def test_single_knot_eval_preserves_ieee_zero_times_infinity(b: float, c: float, d: float) -> None:
    spline = SplineCoeffs(
        x=np.array([0.0, 1.0]),
        y=np.array([1.0, 2.0]),
        n_control=2,
        a=np.array([1.0]),
        b=np.array([b]),
        c=np.array([c]),
        d=np.array([d]),
    )

    scalar_at_knot = spline_single_knot_eval(0.0, spline, 0)
    vector_values = spline_single_knot_eval(np.array([0.0, 1.0]), spline, 0)

    assert np.isnan(scalar_at_knot)
    assert np.isnan(vector_values[0])
    assert vector_values[1] == np.inf


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_akima_spline_coefficients_preserve_input_precision(dtype: type[np.floating]) -> None:
    x, y = _affine_control_points(dtype)
    spline = AkimaSpline(x, y)

    coeff_dtypes = {spline.spline.a.dtype, spline.spline.b.dtype, spline.spline.c.dtype, spline.spline.d.dtype}
    assert coeff_dtypes == {np.dtype(dtype)}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_single_knot_vector_eval_preserves_input_precision_with_typed_spline(dtype: type[np.floating]) -> None:
    spline = _typed_affine_spline(dtype)
    xint = np.array([0.5, 1.5], dtype=dtype)

    assert spline_single_knot_eval(xint, spline, 0).dtype == np.dtype(dtype)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_single_knot_scalar_eval_preserves_input_precision_with_typed_spline(dtype: type[np.floating]) -> None:
    spline = _typed_affine_spline(dtype)

    assert isinstance(spline_single_knot_eval(dtype(0.5), spline, 0), (float, dtype))


@pytest.mark.parametrize('call', [cubic_call_vector, cubic_call_vector_linear, cubic_call])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_vector_call_eval_preserves_input_precision_with_typed_spline(
    dtype: type[np.floating],
    call: Callable[[np.ndarray, SplineCoeffs, int], np.ndarray],
) -> None:
    spline = _typed_affine_spline(dtype)
    xint = np.array([0.5, 1.5], dtype=dtype)

    assert call(xint, spline, 3).dtype == np.dtype(dtype)


@pytest.mark.parametrize('call', [cubic_call_scalar, cubic_call])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_scalar_call_eval_preserves_input_precision_with_typed_spline(
    dtype: type[np.floating],
    call: Callable[[np.floating, SplineCoeffs, int], np.floating | float],
) -> None:
    spline = _typed_affine_spline(dtype)

    assert isinstance(call(dtype(0.5), spline, 3), (dtype, float))
